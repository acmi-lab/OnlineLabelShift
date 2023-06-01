import logging
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset

import utils.label_shift_utils as lsu
import utils.proj_utils as prju
import utils.transf_utils as tsfu

import pickle


logger = logging.getLogger("online_label_shift")

DATA_NUM_LABELS = {
    "synthetic": 3,
    "cifar10": 10,
    "mnist": 10,
    "cinic": 10,
    "fashion": 10,
    "shl": 6,
    "arxiv": 23,
    "eurosat": 10,
}


class Subset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):

        # logger.debug(f"IDx recieved {idx}")
        # logger.debug(f"Indices type {type(self.indices[idx])} value {self.indices[idx]}")
        x = self.dataset[self.indices[idx]]

        if self.transform is not None:
            transformed_img = self.transform(x[0])
            return transformed_img, x[1]
        else:
            return x

    @property
    def y_array(self):
        return self.dataset.y_array[self.indices]

    def __len__(self):
        return len(self.indices)


class ArxivDataset(torch.utils.data.Dataset):
    """Arxiv Dataset.
    Args:
        root_dir (string): Root directory of dataset where dataset file exist.
        transform (callable, optional): A function/transform that takes in
            input title and returns a transformed version (e.g., removing stopwords).
    """

    def __init__(self, root_dir, transform=None, indices=None):

        self.dataset_name = "arxiv"
        self.data_path = f"{root_dir}/arxiv_sets.pkl"
        with open(self.data_path, "rb") as f:
            self.data = pickle.load(f)

        self.title_array = []
        self.y_array = []
        self.time_ids = []

        for time_id in self.data.keys():
            self.title_array.append(self.data[time_id]["title"])
            self.y_array.append(self.data[time_id]["category"].astype(np.int_))
            self.time_ids.append(
                np.array([time_id] * len(self.data[time_id]["category"]))
            )

        self.title_array = np.concatenate(self.title_array, axis=0)
        self.y_array = np.concatenate(self.y_array, axis=0)
        self.time_ids = np.concatenate(self.time_ids, axis=0)
        if indices is None:
            self.indices = np.arange(0, len(self.y_array))
        else:
            self.indices = indices
        self.transform = transform

        self.map_idx = np.load(f"{root_dir}/arxiv_map.npz.npy", allow_pickle=True).item()

        self.y_array = np.array([self.map_idx[y] for y in self.y_array])

    def __getitem__(self, idx):
        """Get title, target, and metadata for data loader.
           Here, the only metadata is the timeid (starting from 0).
        Args:
            index (int): Index
        Returns:
            tuple: (title, target) where target is index of the target class.
        """

        if self.transform is not None:
            title = self.transform(self.title_array[idx])
        else:
            title = self.title_array[idx]
        return title, self.y_array[idx], self.time_ids[idx]

    def __len__(self):
        """Return size of the dataset."""
        return len(self.indices)

    def get_time_split(self, time_ids):
        """
        Args:
            time_ids (list): indinces into time periods.
        Returns:
            Subset dataset which only returns data from
            specific time_ids.
        """
        selected_indices = []
        for time_id in time_ids:
            selected_indices.append(self.indices[self.time_ids == time_id])
        selected_indices = np.concatenate(selected_indices, 0)
        assert len(selected_indices) > 0, "No data found in specified time_id."
        return Subset(dataset=self, indices=selected_indices, transform=self.transform)

    @property
    def num_classes(self):
        return 23
        # return len(np.unique(self.y_array))

    @property
    def num_time_steps(self):
        return len(np.unique(self.time_ids))


def arxiv_split_idx(targets, num_classes, source_frac, seed):
    """
    Returns the indices of the source and target sets
    Input:
        dataset_len: length of the dataset
        source_frac: fraction of the dataset to use as source
        seed: seed for the random number generator
    Output:
        source_idx: indices of the source set
        target_idx: indices of the target set
    """

    np.random.seed(seed)
    idx_per_label = []
    num_per_label = []

    for i in range(num_classes):
        idx_per_label.append(np.where(targets == i)[0])
        num_per_label.append(len(idx_per_label[i]))

    # np.save("arxiv_map.npz", map_idx)
    source_idx = []
    target_idx = []
    for i in range(num_classes):
        source_idx.extend(
            np.random.choice(
                idx_per_label[i], int(source_frac * len(idx_per_label[i])), replace=False
            )
        )
        target_idx.extend(np.setdiff1d(idx_per_label[i], source_idx, assume_unique=True))

    source_idx = np.random.choice(source_idx, int(0.1 * len(source_idx)), replace=False)
    target_idx = np.random.choice(target_idx, int(0.1 * len(target_idx)), replace=False)
    return np.array(source_idx), np.array(target_idx)


class LabelShiftedSubset(Subset):
    def __init__(self, dataset, indices, transform=None, sampling_target_marginals=None):
        super().__init__(dataset, indices, transform)
        self.sampling_target_marginals = sampling_target_marginals

    @property
    def num_labels(self):
        return np.max(self.dataset.y_array) + 1

    @property
    def true_label_marginals(self):
        return lsu.get_label_marginals(self.y_array, self.num_labels)


def split_idx(targets, num_classes, source_frac, seed):
    """
    Returns the indices of the source and target sets
    Input:
            dataset_len: length of the dataset
            source_frac: fraction of the dataset to use as source
            seed: seed for the random number generator
    Output:
            source_idx: indices of the source set
            target_idx: indices of the target set
    """

    np.random.seed(seed)
    idx_per_label = []
    for i in range(num_classes):
        idx_per_label.append(np.where(targets == i)[0])

    source_idx = []
    target_idx = []
    for i in range(num_classes):
        source_idx.extend(
            np.random.choice(
                idx_per_label[i], int(source_frac * len(idx_per_label[i])), replace=False
            )
        )
        target_idx.extend(np.setdiff1d(idx_per_label[i], source_idx, assume_unique=True))

    return np.array(source_idx), np.array(target_idx)


def split_dataset(dataset, num_labels, frac, seed):
    """Returns the two datasets split from dataset.
    The first dataset has fraction frac of the original dataset
    Input:
            dataset_len: length of the dataset
            source_frac: fraction of the dataset to use as source
            seed: seed for the random number generator
    Output:
            source_idx: indices of the source set
            target_idx: indices of the target set
    """
    if frac >= 1.0:
        return dataset, None

    idc_1, idc_2 = split_idx(dataset.y_array, num_labels, source_frac=frac, seed=seed)

    dataset_1 = Subset(dataset, idc_1)
    dataset_2 = Subset(dataset, idc_2)
    return dataset_1, dataset_2


def dataset_with_targets(cls):
    """
    Modifies the dataset class to return target
    """

    def y_array(self):
        return np.array(self.targets).astype(int)

    dst_target = type(cls.__name__, (cls,), {"y_array": property(y_array)})
    return dst_target


def get_datasets(dataname, **params):
    functions = {
        prju.DatasetParameters.CIFAR10_NAME: get_cifar10,
        prju.DatasetParameters.SYNDATA_NAME: get_synthetic,
        prju.DatasetParameters.MNIST_NAME: get_mnist,
        prju.DatasetParameters.FASHION_NAME: get_fashion,
        prju.DatasetParameters.ARXIV_NAME: get_arxiv,
        prju.DatasetParameters.CINIC_NAME: get_cinic,
        prju.DatasetParameters.EUROSAT_NAME: get_eurosat,
        prju.DatasetParameters.SHL_NAME: get_shl,
    }
    if dataname not in functions:
        raise ValueError(f"Dataset {dataname} not supported")
    else:
        return functions[dataname](**params)


def get_arxiv(
    source=True,
    target=False,
    root_path=None,
    transforms=None,
    num_classes=10,
    seed=42,
):
    data_folder = prju.Parameters.get_dataset_path(root_path)
    root_dir = f"{data_folder}/{prju.DatasetParameters.ARXIV_NAME}"

    # Setup default transform
    arxiv_transforms = tsfu.initialize_bert_transform("distilbert-base-uncased")

    dataset = ArxivDataset(root_dir=root_dir, transform=None)
    time_split_dataset = dataset.get_time_split(time_ids=list(range(15)))

    train_indices, val_indices = arxiv_split_idx(
        time_split_dataset.y_array,
        num_classes=dataset.num_classes,
        source_frac=0.8,
        seed=seed,
    )

    val_split = np.random.choice(val_indices, int(0.5 * len(val_indices)), replace=False)

    test_split = np.setdiff1d(val_indices, val_split, assume_unique=True)

    if source:

        source_trainset = Subset(
            time_split_dataset, indices=train_indices, transform=arxiv_transforms
        )

        source_testset = Subset(time_split_dataset, val_split, transform=arxiv_transforms)

        logger.debug(
            f"Size of source data; train {len(source_trainset)} and test {len(source_testset)}"
        )
    if target:
        targetset = Subset(time_split_dataset, test_split, transform=arxiv_transforms)

    datasets = {}
    if source and target:
        datasets["source_train"] = source_trainset
        datasets["source_test"] = source_testset
        datasets["target"] = targetset
    elif source:
        datasets["source_train"] = source_trainset
        datasets["source_test"] = source_testset
    elif target:
        datasets["target"] = targetset

    return datasets


def get_mnist(
    source=True,
    target=False,
    root_path=None,
    transforms=None,
    num_classes=10,
    seed=42,
):
    data_folder = prju.Parameters.get_dataset_path(root_path)
    root_dir = f"{data_folder}/{prju.DatasetParameters.MNIST_NAME}"

    # Setup default transform
    mnist_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    if transforms is None:
        transforms = {
            "source_train": mnist_transforms,
            "source_test": mnist_transforms,
            "target": mnist_transforms,
        }

    MNIST = dataset_with_targets(torchvision.datasets.MNIST)
    if source:
        trainset = MNIST(root=root_dir, train=True, download=True, transform=None)
        source_train_idx, source_test_idx = split_idx(
            trainset.y_array, num_classes, source_frac=0.8, seed=seed
        )

        source_trainset = Subset(
            trainset, source_train_idx, transform=transforms["source_train"]
        )
        source_testset = Subset(
            trainset, source_test_idx, transform=transforms["source_test"]
        )
        logger.debug(
            f"Size of source data; train {len(source_trainset)} and test {len(source_testset)}"
        )
    if target:
        targetset = MNIST(
            root=root_dir, train=False, download=False, transform=transforms["target"]
        )

    datasets = {}
    if source and target:
        datasets["source_train"] = source_trainset
        datasets["source_test"] = source_testset
        datasets["target"] = targetset
    elif source:
        datasets["source_train"] = source_trainset
        datasets["source_test"] = source_testset
    elif target:
        datasets["target"] = targetset

    return datasets


def get_fashion(
    source=True,
    target=False,
    root_path=None,
    transforms=None,
    num_classes=10,
    seed=42,
):
    data_folder = prju.Parameters.get_dataset_path(root_path)
    root_dir = f"{data_folder}/{prju.DatasetParameters.FASHION_NAME}"

    # Setup default transform
    fashion_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )
    if transforms is None:
        transforms = {
            "source_train": fashion_transforms,
            "source_test": fashion_transforms,
            "target": fashion_transforms,
        }

    FashionMNIST = dataset_with_targets(torchvision.datasets.FashionMNIST)
    if source:
        trainset = FashionMNIST(root=root_dir, train=True, download=True, transform=None)
        source_train_idx, source_test_idx = split_idx(
            trainset.y_array, num_classes, source_frac=0.8, seed=seed
        )

        source_trainset = Subset(
            trainset, source_train_idx, transform=transforms["source_train"]
        )
        source_testset = Subset(
            trainset, source_test_idx, transform=transforms["source_test"]
        )
        logger.debug(
            f"Size of source data; train {len(source_trainset)} and test {len(source_testset)}"
        )
    if target:
        targetset = FashionMNIST(
            root=root_dir, train=False, download=False, transform=transforms["target"]
        )

    datasets = {}
    if source and target:
        datasets["source_train"] = source_trainset
        datasets["source_test"] = source_testset
        datasets["target"] = targetset
    elif source:
        datasets["source_train"] = source_trainset
        datasets["source_test"] = source_testset
    elif target:
        datasets["target"] = targetset

    return datasets


def get_cifar10(
    source=True,
    target=False,
    root_path=None,
    transforms=None,
    num_classes=10,
    seed=42,
):
    data_folder = prju.Parameters.get_dataset_path(root_path)
    root_dir = f"{data_folder}/{prju.DatasetParameters.CIFAR10_NAME}"

    # Setup default transform
    if transforms is None:
        transforms = {
            "source_train": tsfu.initialize_transform(
                dataset_name=f"{prju.DatasetParameters.CIFAR10_NAME}"
            ),
            "source_test": tsfu.initialize_transform(
                dataset_name=f"{prju.DatasetParameters.CIFAR10_NAME}"
            ),
            "target": tsfu.initialize_transform(
                dataset_name=f"{prju.DatasetParameters.CIFAR10_NAME}"
            ),
        }

    CIFAR10 = dataset_with_targets(torchvision.datasets.CIFAR10)
    if source:
        trainset = CIFAR10(root=root_dir, train=True, download=True, transform=None)
        source_train_idx, source_test_idx = split_idx(
            trainset.y_array, num_classes, source_frac=0.8, seed=seed
        )

        source_trainset = Subset(
            trainset, source_train_idx, transform=transforms["source_train"]
        )
        source_testset = Subset(
            trainset, source_test_idx, transform=transforms["source_test"]
        )
        logger.debug(
            f"Size of source data; train {len(source_trainset)} and test {len(source_testset)}"
        )
    if target:
        targetset = CIFAR10(
            root=root_dir, train=False, download=True, transform=transforms["target"]
        )

    datasets = {}
    if source and target:
        datasets["source_train"] = source_trainset
        datasets["source_test"] = source_testset
        datasets["target"] = targetset
    elif source:
        datasets["source_train"] = source_trainset
        datasets["source_test"] = source_testset
    elif target:
        datasets["target"] = targetset

    return datasets


def get_eurosat(
    source=True,
    target=False,
    root_path=None,
    transforms=None,
    num_classes=10,
    seed=42,
):
    data_folder = prju.Parameters.get_dataset_path(root_path)
    root_dir = f"{data_folder}/{prju.DatasetParameters.EUROSAT_NAME}"

    # Setup default transform
    if transforms is None:
        transforms = {
            "source_train": tsfu.initialize_transform(
                dataset_name=f"{prju.DatasetParameters.EUROSAT_NAME}",
                target_resolution=64,
            ),
            "source_test": tsfu.initialize_transform(
                dataset_name=f"{prju.DatasetParameters.EUROSAT_NAME}",
                target_resolution=64,
            ),
            "target": tsfu.initialize_transform(
                dataset_name=f"{prju.DatasetParameters.EUROSAT_NAME}",
                target_resolution=64,
            ),
        }

    EuroSAT = dataset_with_targets(torchvision.datasets.EuroSAT)

    if source:
        trainset = EuroSAT(root=root_dir, download=False, transform=None)

        source_idx, target_idx = split_idx(
            trainset.y_array, num_classes, source_frac=0.8, seed=seed
        )

        source_trainset_idx, source_testset_idx = split_idx(
            trainset.y_array[source_idx], num_classes, source_frac=0.8, seed=seed
        )
        source_trainset_idx, source_testset_idx = (
            source_idx[source_trainset_idx],
            source_idx[source_testset_idx],
        )

        source_trainset = Subset(
            trainset, source_trainset_idx, transform=transforms["source_train"]
        )
        source_testset = Subset(
            trainset, source_testset_idx, transform=transforms["source_test"]
        )
        logger.debug(
            f"Size of source data; train {len(source_trainset)} and test {len(source_testset)}"
        )
    if target:
        targetset = Subset(trainset, target_idx, transform=transforms["source_test"])

    datasets = {}
    if source and target:
        datasets["source_train"] = source_trainset
        datasets["source_test"] = source_testset
        datasets["target"] = targetset
    elif source:
        datasets["source_train"] = source_trainset
        datasets["source_test"] = source_testset
    elif target:
        datasets["target"] = targetset

    return datasets


def get_cinic(
    source=True,
    target=False,
    root_path=None,
    transforms=None,
    num_classes=10,
    seed=42,
):
    data_folder = prju.Parameters.get_dataset_path(root_path)
    root_dir = f"{data_folder}/{prju.DatasetParameters.CINIC_NAME}-10"

    # Setup default transform
    if transforms is None:
        transforms = {
            "source_train": tsfu.initialize_transform(
                dataset_name=f"{prju.DatasetParameters.CINIC_NAME}"
            ),
            "source_test": tsfu.initialize_transform(
                dataset_name=f"{prju.DatasetParameters.CINIC_NAME}"
            ),
            "target": tsfu.initialize_transform(
                dataset_name=f"{prju.DatasetParameters.CINIC_NAME}"
            ),
        }

    ImageFolder = dataset_with_targets(torchvision.datasets.ImageFolder)

    if source:

        trainset = ImageFolder(f"{root_dir}/train")

        source_train_idx, source_test_idx = split_idx(
            trainset.y_array, num_classes, source_frac=0.8, seed=seed
        )

        source_trainset = Subset(
            trainset, source_train_idx, transform=transforms["source_train"]
        )
        source_testset = Subset(
            trainset, source_test_idx, transform=transforms["source_test"]
        )
        logger.debug(
            f"Size of source data; train {len(source_trainset)} and test {len(source_testset)}"
        )
    if target:
        targetset = ImageFolder(f"{root_dir}/valid", transform=transforms["target"])

    datasets = {}
    if source and target:
        datasets["source_train"] = source_trainset
        datasets["source_test"] = source_testset
        datasets["target"] = targetset
    elif source:
        datasets["source_train"] = source_trainset
        datasets["source_test"] = source_testset
    elif target:
        datasets["target"] = targetset

    return datasets


def get_shl(
    source=True,
    target=False,
    root_path=None,
    transforms=None,
    num_classes=10,
    seed=42,
):

    data_folder = prju.Parameters.get_dataset_path(root_path)
    root_dir = f"{data_folder}/{prju.DatasetParameters.SHL_NAME}"

    sourceset, targetset, meta_data = get_table_data(root_dir, seed)

    if source:

        source_train_idx, source_test_idx = split_idx(
            sourceset.y_array, num_classes, source_frac=0.8, seed=seed
        )

        source_trainset = Subset(sourceset, source_train_idx, transform=None)

        source_testset = Subset(sourceset, source_test_idx, transform=None)

        logger.debug(
            f"Size of source data; train {len(source_trainset)} and test {len(source_testset)}"
        )

    if target:

        target_testset = targetset

    datasets = {}
    if source and target:
        datasets["source_train"] = source_trainset
        datasets["source_test"] = source_testset
        datasets["target"] = targetset

    elif source:
        datasets["source_train"] = source_trainset
        datasets["source_test"] = source_testset

    elif target:
        datasets["target"] = targetset

    return datasets


def get_table_data(root_dir, seed=42, bbox=False):
    class CustomDataset(Dataset):
        def __init__(self, data, labels, datetime, ori_labels=None):
            self.data = data
            self.labels = labels
            self.datetime = datetime
            self.ori_labels = ori_labels

            if isinstance(self.data, np.ndarray):
                self.data = torch.from_numpy(self.data)
            self.data = self.data.float()

        def __getitem__(self, index):
            pass

        def __len__(self):
            return len(self.data)

        @property
        def y_array(self):
            return self.labels

    class TrainDataset(CustomDataset):
        def __init__(self, data, labels, datetime, ori_labels=None):
            super(TrainDataset, self).__init__(data, labels, datetime, ori_labels)

        def __getitem__(self, index):
            return self.data[index], self.labels[index]

        def fission(self, test_ratio):
            from sklearn.model_selection import train_test_split

            (
                train_data,
                test_data,
                train_labels,
                test_labels,
                train_datetime,
                test_datetime,
            ) = train_test_split(
                self.data,
                self.labels,
                self.datetime,
                test_size=test_ratio,
                shuffle=True,
                random_state=1024,
            )

            return self.__class__(
                train_data, train_labels, test_datetime
            ), self.__class__(test_data, test_labels, test_datetime)

    class TestDataset(CustomDataset):
        def __init__(self, data, labels, datetime, seed):
            super(TestDataset, self).__init__(data, labels, datetime)
            self.rng = np.random.default_rng(seed)
            self.interval = 1
            self.ptr = 0
            print("Online Batch Size: {}".format(self.interval))
            if isinstance(self.labels, np.ndarray):
                self.labels = torch.from_numpy(self.labels)

            self.drop = False

        def __getitem__(self, t):
            self.ptr = t
            X = np.squeeze(self.data[self.ptr : self.ptr + self.interval])
            y = self.labels[self.ptr : self.ptr + self.interval]

            if self.drop:
                idx = self.rng.integers(self.interval)
                X = torch.cat((X[:idx], X[idx + 1 :]), dim=0)
                y = torch.cat([y[:idx], y[idx + 1 :]], dim=0)

            return X, y

    from os.path import isdir, join
    from sklearn import preprocessing
    import pandas as pd
    from sklearn.utils import shuffle

    path = root_dir

    if isdir(path):
        train_data = pd.read_csv(join(path, "train_data.csv"))
        test_data = pd.read_csv(join(path, "test_data.csv"))
        features_train = train_data.drop(columns=["y", "date"]).values
        features_test = test_data.drop(columns=["y", "date"]).values
        labels_train, labels_test = train_data["y"], test_data["y"]
        dates_train, dates_test = train_data["date"], test_data["date"]

        features_test, labels_test, dates_test = shuffle(
            features_test, labels_test, dates_test, random_state=seed
        )

    else:
        data = pd.read_csv(path)
        data = data.sort_values(by="date")

        features = data.drop(columns=["y", "date"]).values
        labels = data["y"].values
        dates = data["date"].values

        total_num = len(labels)
        split_ratio = 0.5
        print("Split ratio: {}".format(split_ratio))
        train_num = int(total_num * split_ratio)
        features_train, features_test = features[:train_num], features[train_num:]
        labels_train, labels_test = labels[:train_num], labels[train_num:]
        dates_train, dates_test = dates[:train_num], dates[train_num:]

    le = preprocessing.LabelEncoder()
    _labels_train = le.fit_transform(labels_train)
    _labels_test = le.transform(labels_test)

    train_set = TrainDataset(
        data=features_train,
        ori_labels=labels_train,
        labels=_labels_train,
        datetime=dates_train,
    )

    if bbox:
        test_set = TrainDataset(
            data=features_test,
            ori_labels=labels_train,
            labels=_labels_test,
            datetime=dates_test,
        )
    else:
        # dataset_type = TestDataset
        test_set = TestDataset(
            data=features_test,
            labels=_labels_test,
            datetime=dates_test,
            seed=seed,
        )

    cls_num = len(np.unique(_labels_train))
    count = torch.bincount(torch.from_numpy(_labels_train), minlength=cls_num)
    priors = count / count.sum()

    info = {
        "cls_num": cls_num,
        "dim": features_train.shape[1],
        "init_priors": priors,
    }

    return train_set, test_set, info


class SyntheticDataset(torch.utils.data.Dataset):
    """The synthetic dataset generated as described in https://arxiv.org/abs/2207.02121"""

    def __init__(self, root_dir, source=True):
        self.root_dir = Path(root_dir)
        self.source = source
        if self.source:
            self.datapath = self.root_dir / "source_data.npz"
        else:
            self.datapath = self.root_dir / "target_data.npz"
        self.data = np.load(self.datapath)
        self.X = torch.from_numpy(self.data["X"]).to(torch.float32)
        self.y = torch.from_numpy(self.data["y"])

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.X[:, idx], self.y[idx]

    @property
    def y_array(self):
        return self.data["y"]


def get_synthetic(
    source=True,
    target=False,
    root_path="./",
    transforms=None,
    num_classes=3,
    seed=17,
):
    data_folder = prju.Parameters.get_dataset_path(root_path)
    root_dir = f"{data_folder}/{prju.DatasetParameters.SYNDATA_NAME}"

    datasets = {}
    if source:
        trainset = SyntheticDataset(root_dir=root_dir, source=True)
        source_train_idx, source_test_idx = split_idx(
            trainset.y_array, num_classes, source_frac=0.8, seed=seed
        )
        source_trainset = Subset(trainset, source_train_idx)
        source_testset = Subset(trainset, source_test_idx)
        logger.debug(
            f"Size of source data: train {len(source_trainset)} and test {len(source_testset)}"
        )
        datasets["source_train"] = source_trainset
        datasets["source_test"] = source_testset

    if target:
        targetset = SyntheticDataset(root_dir=root_dir, source=False)
        datasets["target"] = targetset

    return datasets
