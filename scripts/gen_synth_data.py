"""Generate synthetic data as described in https://arxiv.org/abs/2207.02121
"""
import argparse
import logging
from pathlib import Path

import init_path
import numpy as np
from init_path import parent_path

import utils.data_utils as datu
import utils.label_shift_utils as lsu
import utils.misc_utils as mscu
import utils.proj_utils as prju

SYNDATA_NAME = prju.DatasetParameters.SYNDATA_NAME
DEFAULTS = prju.DatasetParameters.dataset_defaults[SYNDATA_NAME]


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-classes", type=int, default=DEFAULTS["num_classes"])
    parser.add_argument("--num-dimensions", type=int, default=DEFAULTS["num_dimensions"])
    parser.add_argument(
        "--num-source-data", type=int, default=DEFAULTS["num_source_data"]
    )
    parser.add_argument(
        "--num-target-data", type=int, default=DEFAULTS["num_target_data"]
    )
    parser.add_argument(
        "--gaussian-centre-distance",
        type=float,
        default=DEFAULTS["gaussian_centre_distance"],
    )
    parser.add_argument(
        "--gaussian-variance", type=float, default=DEFAULTS["gaussian_variance"]
    )
    parser.add_argument("--seed", type=int, default=21)
    return parser


def generate_data(
    num_classes,
    num_dimensions,
    num_source_data,
    num_target_data,
    gaussian_centre_distance,
    gaussian_variance,
    datafolder,
    logger: logging.Logger,
):
    # Generate gaussian centres
    centres = np.zeros((num_dimensions, num_classes))
    for i in range(num_classes):
        centre = np.random.randn(num_dimensions)
        centre = centre / np.linalg.norm(centre) * gaussian_centre_distance
        centres[:, i] = centre
        logger.info(f"Centre {i}: {centre}")
        logger.debug(f"Centre length {i}: {np.linalg.norm(centre)}")

    # Generate data
    nsource = num_source_data // num_classes
    ntarget = num_target_data // num_classes
    source_data = np.zeros((num_dimensions, nsource * num_classes))
    source_labels = np.zeros(nsource * num_classes).astype(int)
    target_data = np.zeros((num_dimensions, ntarget * num_classes))
    target_labels = np.zeros(ntarget * num_classes).astype(int)
    for i in range(num_classes):
        centre = centres[:, i]
        cov = np.identity(num_dimensions) * gaussian_variance
        cur_source_data = np.random.multivariate_normal(
            mean=centre, cov=cov, size=nsource
        )
        cur_target_data = np.random.multivariate_normal(
            mean=centre, cov=cov, size=ntarget
        )
        source_data[:, i * nsource : (i + 1) * nsource] = cur_source_data.T
        source_labels[i * nsource : (i + 1) * nsource] = i
        target_data[:, i * ntarget : (i + 1) * ntarget] = cur_target_data.T
        target_labels[i * ntarget : (i + 1) * ntarget] = i

    # Save data
    source_filepath = str(datafolder / "source_data")
    target_filepath = str(datafolder / "target_data")
    logger.debug(f"Saving data to {source_filepath} and {target_filepath}")
    np.savez(source_filepath, X=source_data, y=source_labels)
    np.savez(target_filepath, X=target_data, y=target_labels)


if __name__ == "__main__":
    # Setup data folder
    datafolder = (
        Path(prju.Parameters.get_dataset_path(root_path=parent_path)) / SYNDATA_NAME
    )
    datafolder.mkdir(parents=True, exist_ok=True)
    log_filename = str(datafolder / "details.txt")

    # Setup log
    logging.basicConfig(format="%(asctime)s %(message)s")
    logger = logging.getLogger("online_label_shift")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(filename=log_filename, mode="w"))

    # Get arguments
    args = get_parser().parse_args()
    logger.info(f"Experiment parameters: {vars(args)}")

    mscu.set_seed(args.seed)
    generate_data(
        num_classes=args.num_classes,
        num_dimensions=args.num_dimensions,
        num_source_data=args.num_source_data,
        num_target_data=args.num_target_data,
        gaussian_centre_distance=args.gaussian_centre_distance,
        gaussian_variance=args.gaussian_variance,
        datafolder=datafolder,
        logger=logger,
    )

    datasets = datu.get_synthetic(source=True, target=True, root_path=parent_path)
    logger.info(f"Number of source train data {len(datasets['source_train'])}")
    logger.info(
        f"Label distribution {lsu.get_label_marginals(datasets['source_train'].y_array)}"
    )
    logger.info(f"First source train data {datasets['source_train'][0]}")

    logger.info(f"Number of source test data {len(datasets['source_test'])}")
    logger.info(
        f"Label distribution {lsu.get_label_marginals(datasets['source_test'].y_array)}"
    )

    target_dataset = datu.SyntheticDataset(root_dir=datafolder, source=False)
    logger.info(f"Number of target data {len(target_dataset)}")
    logger.info(f"Label distribution {lsu.get_label_marginals(target_dataset.y_array)}")
    logger.info(f"First target data {target_dataset[0]}")
