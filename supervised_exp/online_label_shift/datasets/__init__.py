from online_label_shift.datasets.get_dataset import *

benchmark_datasets = [
    'cifar10',
    'cifar100',
    'fmow',
    'mnist',
    'fashion',
    'arxiv',
    'precipitation',
    'cifar10-5m',
    'living17',
    'civilcomments',
    'retiring_adult',
    'mimic_readmission',
]

supported_datasets = benchmark_datasets 

dataset_map = { 
    "cifar10" : get_cifar10, 
    "cifar100" : get_cifar100,
    "mnist": get_mnist,
    # 'living17': get_living17,
    # 'fmow': get_fmow,
    # 'civilcomments': get_civilcomments,
    # 'amazon': get_amazon, 
    # 'retiring_adult': get_retiring_adult,
    # 'mimic_readmission': get_mimic_readmission,
}

def get_dataset(dataset, root_dir = None, seed=42):
    """
    Returns the appropriate dataset
    Input:
        dataset: name of the dataset
        root_dir: root directory of the dataset
    Output:
        dataset: labeled dataset
    """

    return dataset_map[dataset](root_dir, seed)


