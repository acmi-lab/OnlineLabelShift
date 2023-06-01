from pathlib import Path

import torch


class Parameters:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = "data"
    model_path = "model"
    result_path = "results"
    plt_path = "plots"

    @classmethod
    def get_dataset_path(cls, root_path="./"):
        path = Path(root_path) / cls.dataset_path
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    @classmethod
    def get_model_path(cls, root_path="./"):
        path = Path(root_path) / cls.model_path
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    @classmethod
    def get_result_path(cls, root_path="./"):
        path = Path(root_path) / cls.result_path
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    @classmethod
    def get_plot_path(cls, root_path="./"):
        path = Path(root_path) / cls.plt_path
        path.mkdir(parents=True, exist_ok=True)
        return str(path)


class DatasetParameters:
    CIFAR10_NAME = "cifar10"
    SYNDATA_NAME = "synthetic"
    MNIST_NAME = "mnist"
    ARXIV_NAME = "arxiv"
    FASHION_NAME = "fashion"
    EUROSAT_NAME = "eurosat"
    CINIC_NAME = "cinic"
    SHL_NAME = "shl"

    supported_datasets = [CIFAR10_NAME, SYNDATA_NAME, MNIST_NAME]
    dataset_defaults = {
        MNIST_NAME: {
            "num_classes": 10,
            "class_names": [
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
            ],
        },
        CIFAR10_NAME: {
            "num_classes": 10,
            "class_names": [
                "plane",
                "car",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ],
        },
        SYNDATA_NAME: {
            "num_classes": 3,
            "class_names": [
                "class 1",
                "class 2",
                "class 3",
            ],
            "num_dimensions": 12,
            "num_source_data": int(6e4),
            "num_target_data": int(1.2e4),
            "gaussian_centre_distance": 1.0,
            "gaussian_variance": 0.215,
        },
    }


class ModelParameters:
    # Map (data_name, model_name) to ROGD's lipschitz constant
    rogd_lipschitz = {
        ("synthetic", "logreg"): 4.110312455268849,
        ("mnist", "fcn"): 0.8526684338308601,
        ("mnist", "fcn_early"): 0.6963140055404636,
        ("mnist", "randforest"): 0.6416969719989557,
        ("cifar10", "resnet18"): 0.6737237785512858,
        ("cifar10", "fcn"): 1.0,
        ("cifar10", "randforest"): 0.6977292003450563,
        ("cinic", "resnet18"): 0.8649583670433504,
        ("eurosat", "resnet18"): 1.9558086453186425,
        ("fashion", "mlp"): 0.6786518583982573,
        ("shl", "mlp"): 1.6636829369691637,
        ("arxiv", "bert"): 0.7263109745596158,
    }
