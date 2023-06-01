import logging

import numpy as np
import torch
import torchvision

from online_label_shift.datasets.data_utils import *

logger = logging.getLogger("label_shift")

def get_cifar10(root_dir, seed,  num_classes=10):
    
    root_dir = f"{root_dir}/cifar10"

    CIFAR10 = dataset_with_targets(torchvision.datasets.CIFAR10)

    trainset = CIFAR10(root=root_dir, train=True, download=False, transform=None)
    
    testset = CIFAR10(root=root_dir, train=False, download=False, transform=None)

    dataset = CustomConcatDataset((trainset, testset))
    logger.debug(f"Size of dataset {len(dataset)}")
    
    train_idx, val_idx = split_idx(dataset.y_array, num_classes=num_classes, source_frac = 59500.0/60000, seed=seed)

    train_set = Subset(dataset, train_idx)
    
    val_set = Subset(dataset, val_idx)

    logger.info(f"Size of train set {len(train_set)} and val set {len(val_set)}")

    dataset = {}
    
    dataset["train"] = train_set
    dataset["val"] = val_set
    
    return dataset


def get_cifar100(root_dir, seed, num_classes=100):
        
    root_dir = f"{root_dir}/cifar100"

    CIFAR100 = dataset_with_targets(torchvision.datasets.CIFAR100)

    trainset = CIFAR100(root=root_dir, train=True, download=False, transform=None)
    
    testset = CIFAR100(root=root_dir, train=False, download=False, transform=None)

    dataset = CustomConcatDataset((trainset, testset))
    logger.debug(f"Size of dataset {len(dataset)}")
    
    train_idx, val_idx = split_idx(dataset.y_array, num_classes=num_classes, source_frac = 59500.0/60000, seed=seed)

    train_set = Subset(dataset, train_idx)
    
    val_set = Subset(dataset, val_idx)

    logger.info(f"Size of train set {len(train_set)} and val set {len(val_set)}")

    dataset = {}
    
    dataset["train"] = train_set
    dataset["val"] = val_set
    
    return dataset

def get_mnist(root_dir, seed, num_classes=100):
    
    root_dir = f"{root_dir}/mnist"

    MNIST = dataset_with_targets(torchvision.datasets.MNIST)

    trainset = MNIST(root=root_dir, train=True, download=False, transform=None)
    
    testset = MNIST(root=root_dir, train=False, download=False, transform=None)

    dataset = CustomConcatDataset((trainset, testset))
    logger.debug(f"Size of dataset {len(dataset)}")

    train_idx, val_idx = split_idx(dataset.y_array, num_classes=num_classes, source_frac = 59500.0/60000, seed=seed)

    train_set = Subset(dataset, train_idx)
    
    val_set = Subset(dataset, val_idx)

    logger.info(f"Size of train set {len(train_set)} and val set {len(val_set)}")

    dataset = {}
    
    dataset["train"] = train_set
    dataset["val"] = val_set
    
    return dataset