import argparse
import copy
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from online_label_shift.datasets import *
from online_label_shift.helper import train 
from online_label_shift.label_shift_utils import *
from online_label_shift.models.initializer import initialize_model
from online_label_shift.transforms import initialize_transform
from online_label_shift.utils import (ParseKwargs, ResultsLogger,
                                     initialize_wandb, log_config,
                                     parse_bool, set_seed)

try:
    import wandb
except Exception as e:
    pass

logFormatter = logging.Formatter('%(asctime)s, [%(levelname)s, %(filename)s:%(lineno)d] %(message)s')

logger = logging.getLogger("label_shift")
logger.setLevel(logging.DEBUG)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)


def get_parser():
    ''' Arg defaults are filled in according to configs/ '''
    parser = argparse.ArgumentParser()
    
    # AWS cluster params

    # Required arguments
    parser.add_argument('-d', '--dataset', choices=supported_datasets, required=True)
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in the dataset.') 
    parser.add_argument('--root_dir', required=True,
                        help='The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--default_normalization', type=parse_bool, const=True, nargs='?', help='Default normalization is applied to the data.')
    parser.add_argument('--mean', type=float, nargs="+", help='Mean of the dataset.')
    parser.add_argument('--std', type=float, nargs="+", help='Std of the dataset.')

    # Online Label Shift params
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to use for online label shift.')
    parser.add_argument('--num_time_steps', type=int, default=1000, help='Number of time steps for the online problem')
    parser.add_argument('--alpha', type=float, default = 10.0, help='Alpha for generating initial and final distributions')
    parser.add_argument('--label_shift_type', type=str, required=True, help='Type of label shift to simulate.', choices=SHIFT_FUNCTIONS.keys())  
    parser.add_argument('--label_shift_kwargs',  nargs='*', action=ParseKwargs, default={}, help='Additional keyword arguments to pass to the label shift function.') 
    # parser.add_argument('--estimate_base_method', type=str, default="RLLS", help='Estimation method for label shift')
    # parser.add_argument('--online_method', type=str, default="None", help='Estimation method for online label shift estimation', choices=["None", "FLH", "FLH-FTL", "UOGD"])
    
    # Model
    parser.add_argument('--model', default = "resnet18", help='Model architecture to use.')
    parser.add_argument('--pretrained', type=parse_bool, const=True, nargs='?', help='If true, load pretrained model.')
    parser.add_argument('--pretrained_path', default="./pretrained_models/resnet18_imagenet32.pt", type=str, help='Specify a path to pretrained model weights')
    parser.add_argument('--calibrate', type=parse_bool, const=True, nargs='?', help='If true, load pretrained model.')

    # Transforms
    parser.add_argument('--transform', default = "image_base", help='Transforms to apply to the data.')
    parser.add_argument('--additional_train_transform',  help='Optional data augmentations to layer on top of the default transforms.')
    parser.add_argument('--target_resolution', type=int,  help='The input resolution that images will be resized to before being passed into the model. For example, use --target_resolution 224 for a standard ResNet.')
    parser.add_argument('--resize_resolution', type=int)
    parser.add_argument('--max_token_length', type=int, default=512, help='Maximum number of tokens in a sentence.')
    parser.add_argument('--randaugment_n', type=int, default=2, help='Number of RandAugment transformations to apply.')

    # Objective
    parser.add_argument('--loss_function', type=str, default='cross_entropy')

    # Algorithm
    parser.add_argument('--algorithm', required=True, default="RS-CT", help='Algorithm to use.', choices=["RS-CT", "RS-RT", "CT", "RT"])    
    parser.add_argument('--optimizer', default='SGD', help='Optimizer to use.')
    # parser.add_argument('--num_iterations', type=int, const=100, help='Number of iteration to train for online')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to max train')
    parser.add_argument('--check_acc', type=int, default=3, help='Number of epochs to max train without accuracy increase')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--optimizer_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for optimizer initialization passed as key1=value1 key2=value2')
    
    # Scheduler
    parser.add_argument('--scheduler')
    parser.add_argument('--scheduler_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for scheduler initialization passed as key1=value1 key2=value2')
    
    # Misc
    parser.add_argument('--device', type=int, nargs='+', default=[0])
    parser.add_argument('--log_dir', default='./logs_submission/', type=str)
    parser.add_argument('--progress_bar', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for the data loader.')

    # Weights & Biases
    parser.add_argument('--use_wandb', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--wandb_api_key_path', type=str,
                        help="Path to Weights & Biases API Key. If use_wandb is set to True and this argument is not specified, user will be prompted to authenticate.")
    parser.add_argument('--wandb_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for wandb.init() passed as key1=value1 key2=value2')

    return parser

def main(config):

    # Initialize logs
    config.log_dir = f"{config.log_dir}/{config.dataset}_shift_type:{config.label_shift_type}_seed:{config.seed}/train:{config.algorithm}_calibrate:{config.calibrate}/lr:{config.lr}_wd:{config.weight_decay}_bs:{config.batch_size}_opt:{config.optimizer}/"
                        
    if os.path.exists(f"{config.log_dir}/finish.txt"): 
        logger.info("The run already existed before ....")
        sys.exit()

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

        logger.info("Logging directory created .... ")
    else: 
        logger.info("Logging directory already exists .... ")


    # Set up logging    
    fileHandler = logging.FileHandler("{0}/{1}".format(config.log_dir, "run.log"), mode='w')


    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)


    # Set device
    if torch.cuda.is_available():
        
        device_count = torch.cuda.device_count()
        if len(config.device) > device_count:
            raise ValueError(f"Specified {len(config.device)} devices, but only {device_count} devices found.")

        config.use_data_parallel = len(config.device) > 1
        device_str = ",".join(map(str, config.device))
        os.environ["CUDA_VISIBLE_DEVICES"] = device_str
        config.device = torch.device("cuda")
    else:
        config.use_data_parallel = False
        config.device = torch.device("cpu")


    # Record config
    logger.info("Config:")
    log_config(config, logger)

    # Set random seed
    set_seed(config.seed)

    # Transforms
    data_transforms = {}

    logger.info("Loading transforms...")
    
    data_transforms["train"] = initialize_transform(
        transform_name=config.transform,
        config=config,
        additional_transform_name=config.additional_train_transform,
        is_training=True, 
        model_name=config.model)

    data_transforms["val"] = initialize_transform(
        transform_name=config.transform,
        config=config,
        is_training=False, 
        model_name=config.model)

    logger.info("Done loading transforms.")

    # Data
    logger.info("Loading data...")

    dataset = get_dataset(
        dataset=config.dataset,
        root_dir=config.root_dir,
        seed = config.seed) 

    logger.info("Done loading data.")

    logger.info("Generating online shifts ... ")
    
    # Get online shifts 
    label_shifts = generate_shifts(
        SHIFT_FUNCTIONS[config.label_shift_type], 
        config.label_shift_kwargs, 
        config.num_time_steps, 
        num_classes=config.num_classes,
        seed=config.seed, 
        alpha=config.alpha)

    # Get datasets    
    generated_shifts_idx = generate_online_shifts(
        dataset["train"].y_array, 
        label_shifts, 
        num_class=config.num_classes, 
        num_samples=config.num_samples, 
        seed=config.seed)
    
    logger.info("Done generating online datasets.")
    
    results_logger = ResultsLogger(
        os.path.join(config.log_dir, 'eval.csv'), mode='w', use_wandb=config.use_wandb
    )

    if config.use_wandb:
        initialize_wandb(config)

    logger.info("Initialize the model...")
    
    model = initialize_model(config.model, config.dataset, config.num_classes, featurize=False, pretrained=False)

    model = model.to(config.device)
    
    train(
        dataset = dataset,
        model = model,
        transforms = data_transforms,
        shift_idx = generated_shifts_idx, 
        results_logger=results_logger,
        config=config
    )
    with open(f"{config.log_dir}/finish.txt", "w") as f: 
        f.write("Done")
    
    if config.use_wandb:
        wandb.finish()

    results_logger.close()

    return config.log_dir

if __name__=='__main__':

    parser = get_parser()
    args = parser.parse_args()
    main(args)
