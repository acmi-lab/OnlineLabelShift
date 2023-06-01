import copy
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import calibration as cal
from online_label_shift.datasets import *
from online_label_shift.models.initializer import *
from online_label_shift.label_shift_utils import *
from online_label_shift.optimizer import *
from online_label_shift.utils import (InfiniteDataIterator, collate_list,
                                     detach_and_clone, load, move_to,
                                     save_model)
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm

logger = logging.getLogger("label_shift")

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def train(dataset, model, transforms, shift_idx, results_logger, config):
   
    logger.info("Training model ...")
    
    dataset["val"] = add_transform(dataset["val"], transforms["val"])
    
    val_dataloader = DataLoader(
                        dataset["val"], 
                        batch_size=config.batch_size,
                        num_workers=config.num_workers,
                        pin_memory=True,)
    
    source_marginal = np.array([1/config.num_classes]*config.num_classes)
    
    estimation_methods = {
        "RLLS": RegularizedShiftEstimator(source_marginal=source_marginal, num_classes=config.num_classes, n_train = len(dataset["val"])), 
        "BBSE": BlackBoxShiftEstimator(source_marginal=source_marginal), 
        "FLH-FTL": FollowLeadingHistoryFollowTheLeaderEstimator(source_marginal=source_marginal, num_labels=config.num_classes), 
        "FTFWH": FollowTheFixedWindowHistoryEstimator(source_marginal=source_marginal, window_size=100), 
        "FTH": FollowTheHistoryEstimator(source_marginal=source_marginal),
    }
    
    if len(shift_idx) < config.num_time_steps:
        logger.info("Number of time steps is larger than the number of shifts. Reducing number of time steps to {}".format(len(shift_idx)))
        config.num_time_steps = len(shift_idx)
    
    # if "oracle" in config.algorithm:
        
    #     all_idx = np.concatenate(shift_idx, axis=0)
        
    #     trainset = Subset(dataset["train"], all_idx, transform=transforms["train"])
        
    #     if "RS" in config.algorithm:
    #         trainloader =  rebalance_loader(trainset, config, use_true_target = True)
    #         train_marginal = np.array([1.0/config.num_classes]*config.num_classes)
        
    #     else: 
    #         trainloader =  DataLoader(trainset, 
    #                     batch_size = config.batch_size, 
    #                     shuffle=True,
    #                     num_workers = config.num_workers, 
    #                     pin_memory = True)
            
    #         train_marginal = calculate_marginal(trainset.targets, config.num_classes)
        
    #     # TODO: add oracle training here
    #     logger.info("Simulating online label shift ... ")
    #     optimizer = initialize_optimizer(model, config)
        
    #     train_one_time(model, optimizer, trainloader, val_dataloader, results_logger, config)
        
    #     for i in range(1, config.num_time_steps):
            
    #         test_online_learner(model, testloader, val_dataloader, config, train_marginal, estimation_methods, results_logger) 
             
        
    # else: 
    logger.info("Simulating online label shift ... ")
    
    optimizer = initialize_optimizer(config, model)
    results = None
    
    for i in range(1, config.num_time_steps):
        
        logger.info("Time step {} ...".format(i))
        
        idx_till_i = np.concatenate(shift_idx[:i], axis=0)
        trainset = Subset(dataset["train"], idx_till_i, transform=transforms["train"])
        
        if "RS" in config.algorithm:
            trainloader =  rebalance_loader(trainset, config, use_true_target = True)
            train_marginal = np.array([1.0/config.num_classes]*config.num_classes)
            
            val_i_dataloader = val_dataloader
        else: 
            trainloader =  DataLoader(trainset,
                        batch_size = config.batch_size, 
                        shuffle=True,
                        num_workers = config.num_workers, 
                        pin_memory = True)

            train_marginal = calculate_marginal(trainset.y_array, config.num_classes)
            val_i_dataloader = rebalance_loader(dataset["val"], config, use_true_target = True, label_marginal=train_marginal)
        
        if "CT" in config.algorithm:
            
            train_one_time(model, optimizer, trainloader, val_i_dataloader, config)
            
        elif "RT" in config.algorithm:
            model = initialize_model(config.model, config.dataset, config.num_classes, featurize=False, pretrained=False)
            model = model.to(config.device)
            
            optimizer = initialize_optimizer(config, model)
                        
            train_one_time(model, optimizer, trainloader, val_i_dataloader, config)
            
        else: 
            raise NotImplementedError("Unknown algorithm {}".format(config.algorithm))                          
        
        
        if config.calibrate:
            cal = calibrate_model(model, val_i_dataloader, config)
        else: 
            cal = None
            
        testset = Subset(dataset["train"], shift_idx[i], transform=transforms["val"])
        testloader = DataLoader(testset, 
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    pin_memory=True)
        

        results = test_online_learner(model, testloader, val_dataloader, config, train_marginal, estimation_methods, results, i, cal) 
        
        logger.info(f"Results at step {i}: {results}")
        results_logger.log(results)
        logger.info(f"Time step {i} finished.")

def train_one_time(model, optimizer, trainloader, val_dataloader, config):
    
    check_acc = config.check_acc
    acc = []
    models = []
    
    for i in range(config.num_epochs): 
        logger.info(f"Epoch {i}")
        train_epoch(model, optimizer, trainloader, config)
        
        acc.append(test_epoch(model, val_dataloader, config))
        models.append(copy.deepcopy(model))
        
        if len(acc) < check_acc:
            pass
        else: 
            j = np.argmax(acc)
            if j == 0: 
                model.load_state_dict(models[0].state_dict())
                logger.info("Early stopping at epoch {}".format(i))
                break
            else:
                old_model = models.pop(0)
                del old_model
                acc.pop(0)
    
    return model    

def test_online_learner(model, testloader, val_dataloader, config, train_marginal, estimation_methods, prev_results, step, calibrator=None):
    
    pred_logits, test_targets = infer_predictions(model, testloader, config)
    
    if calibrator is not None:
        pred_logits= calibrator.calibrate(pred_logits)
    
    pred_probs = softmax(pred_logits, axis=1)
    
    test_marginal = calculate_marginal(test_targets, config.num_classes)
    
    source_logits, source_targets = infer_predictions(model, val_dataloader, config)
    
    if calibrator is not None:
        source_logits= calibrator.calibrate(source_logits)
    
    
    source_probs = softmax(source_logits, axis=1)
    
    confusion_matrix = confusion_matrix_probabilistic( source_targets, source_probs, config.num_classes)
    
    pred_marginals = np.mean(pred_probs, axis=0)
    
    results = {}
    
    results["time"] = step
    for method in estimation_methods: 
        
        if method in ("BBSE", "RLLS"): 
            estimated_marginal = estimation_methods[method].get_marginal_estimate(pred_marginals, confusion_matrix)
        else: 
            estimated_marginal = estimation_methods[method].get_marginal_estimate(test_marginal)

        results[f"{method}_acc"] = im_reweight_acc(estimated_marginal/train_marginal, pred_probs, test_targets)
        results[f"{method}_est_err"] = estimation_err(estimated_marginal, test_marginal)
        
        if prev_results is None:
            results[f"agg_{method}_acc"] = results[f"{method}_acc"]
            results[f"agg_{method}_est_err"] = results[f"{method}_est_err"]
        else: 
            results[f"agg_{method}_acc"] = (prev_results[f"agg_{method}_acc"]*(step-1) + results[f"{method}_acc"])/ step
            results[f"agg_{method}_est_err"] = (prev_results[f"agg_{method}_est_err"]*(step-1) + results[f"{method}_est_err"])/step
    
    results["base_acc"] = get_acc(pred_probs, test_targets)
    results["oracle_acc"] = im_reweight_acc(test_marginal/train_marginal, pred_probs, test_targets)
    
    if prev_results is None:
        results["agg_base_acc"] = results["base_acc"]
        results["agg_oracle_acc"] = results["oracle_acc"]
    else: 
        results["agg_base_acc"] = (prev_results["agg_base_acc"]*(step-1) + results["base_acc"])/ step
        results["agg_oracle_acc"] = (prev_results["agg_oracle_acc"]*(step-1) + results["oracle_acc"])/ step
    
    return results

def calibrate_model(model, val_dataloader, config): 
    
    calibrator = cal.VectorScaling(num_label=config.num_classes, bias=True, device = config.device)
    
    logits, targets = infer_predictions(model, val_dataloader, config)
    
    calibrator.fit(logits, targets)
    
    return calibrator
    

def infer_predictions(model, loader, config):
    """
    Simple inference loop that performs inference using a model (not algorithm) and returns model outputs.
    Compatible with both labeled and unlabeled WILDS datasets.
    """
    model.eval()
    logits = []
    targets = []
    iterator = tqdm(loader) if config.progress_bar else loader
    
    with torch.no_grad(): 
        for batch in iterator:
            x = batch[0]
            x = move_to(x, config.device)
            # x = x.to(config.device)
            output = model(x)
            logits.append(detach_and_clone(output).cpu().numpy())
            targets.append(batch[1].numpy())
            
    return np.concatenate(logits, axis=0), np.concatenate(targets, axis=0)


def train_epoch(model, optimizer, trainloader, config):
    
    model.train()
    iterator = tqdm(trainloader) if config.progress_bar else trainloader
    criterion = nn.CrossEntropyLoss()

    for batch in iterator: 
        x,y = batch[0], batch[1]
        x = move_to(x, config.device)
        y = move_to(y, config.device)
    
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        
        loss.backward()
        optimizer.step()
    
def test_epoch(model, testloader, config):
    
    pred_probs, test_targets = infer_predictions(model, testloader, config)
    
    return get_acc(pred_probs, test_targets) 
        

def save_model_if_needed(algorithm, epoch, config):

    if config.save_every is not None and (epoch) % config.save_every == 0:
        save_model(algorithm, epoch, f'{config.log_dir}/epoch:{epoch}_model.pth')

    if config.save_last:
        save_model(algorithm, epoch, f'{config.log_dir}/epoch:last_model.pth')



def rebalance_loader(dataset, config, use_true_target = True, label_marginal = None): 

    if use_true_target:
        target = np.array(dataset.y_array)
    
    class_counts = np.bincount(target, minlength = config.num_classes)
    max_count = max(class_counts)
    class_counts_proxy = class_counts +  1e-8
    class_weights = max_count / class_counts_proxy

    class_weights[class_counts == 0] = 0

    # import pdb; pdb.set_trace()

    if label_marginal is not None: 
        class_weights = class_weights * label_marginal
    
    weights = class_weights[target]
    sampler = WeightedRandomSampler(weights, len(weights))

    loader = DataLoader(dataset, 
        batch_size = config.batch_size, 
        sampler = sampler, 
        num_workers = config.num_workers, 
        pin_memory = True, 
    )

    return loader
