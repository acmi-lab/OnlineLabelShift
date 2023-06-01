"""This scripts collect ols results from ROOT/results
"""
import argparse
import json
import os

import init_path
import numpy as np

import utils.label_shift_utils as lsu
import utils.misc_utils as mscu
import utils.saveload_utils as slu
from utils.data_utils import DATA_NUM_LABELS
from utils.model_utils import SUPPORTED_LOCAL_SHIFT_ESTIMATORS

DEFAULT_DATA_NAMES = [
    "synthetic",
    "mnist",
    "cifar10",
    # "eurosat",
    # "fashion",
    # "arxiv",
]
DEFAULT_SHIFT_NAMES = [
    "monotone",
    # "square",
    "bernouli",
    # "sinusoidal",
]
DATA_MODELS = {
    "synthetic": "logreg",
    "mnist": "fcn_early",
    # "mnist": "randforest",
    "cifar10": "resnet18",
    # "cifar10": "randforest",
    "eurosat": "resnet18",
    "fashion": "mlp",
    "shl": "mlp",
    "arxiv": "bert",
}
DEFAULT_METHODS = [
    "base",
    # "ofc",
    # "RW_SIMP-LOCAL_BBSE",
    # "RW_FTH_BBSE",
    "RW_FTFWH_BBSE",
    "RW_ROGD_BBSE",
    "UOGD_BBSE",
    # "ATLAS_BBSE",
    "RW_FLH-FTL_BBSE",
    "LIN-RETRAIN_FLH-FTL_BBSE",
]


def get_default_shift_parameters(num_labels):
    marg1 = np.ones(num_labels) / num_labels
    marg2 = np.zeros(num_labels)
    marg2[0] = 1.0

    return marg1, marg2


DATA_MARGINALS = {
    data_name: get_default_shift_parameters(DATA_NUM_LABELS[data_name])
    for data_name in DATA_NUM_LABELS
}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data-names",
        type=str,
        required=False,
        default=" ".join(DEFAULT_DATA_NAMES),
    )
    parser.add_argument(
        "-s",
        "--shift-names",
        type=str,
        required=False,
        default=" ".join(DEFAULT_SHIFT_NAMES),
    )
    parser.add_argument(
        "-m",
        "--method-names",
        type=str,
        required=False,
        default=" ".join(DEFAULT_METHODS),
    )
    parser.add_argument(
        "-n", "--num-online-samples", default=10, type=int, required=False
    )
    parser.add_argument("-t", "--total-time", default=1000, type=int, required=False)
    parser.add_argument("-l", "--log-level", default="INFO", required=False)
    parser.add_argument("--source-test-ratio", default=1.0, type=float, required=False)
    parser.add_argument(
        "--use-source-test", default=True, type=mscu.boolean, required=False
    )
    parser.add_argument(
        "--calibration-type", default="ts", type=str, required=False, choices=["ts", "vs"]
    )
    parser.add_argument(
        "--shift-estimator-type",
        default="bbse",
        type=str,
        required=False,
        choices=SUPPORTED_LOCAL_SHIFT_ESTIMATORS,
    )
    parser.add_argument(
        "--use-current-marginal-estimate",
        default=False,
        type=mscu.boolean,
        required=False,
    )

    return parser


def collect_results(result_folder, logger):
    # Get results filepaths
    experiment_folders = []
    try:
        for filename in os.listdir(result_folder):
            experiment_folders.append(str(result_folder) + f"/{filename}")
    except Exception as e:
        logger.info(
            f"Folder {result_folder} cannot be read."
            f" Error {e.__class__.__name__} - {e}"
        )

    # Load results
    result_jsons = []
    for folder in experiment_folders:
        result_filename = folder + "/results.json"
        try:
            with open(result_filename, "r") as f:
                cur_json = json.load(f)
                cur_json["detailed_results"] = {}
            for method_name in os.listdir(folder):
                subfolder = f"{folder}/{method_name}"
                if os.path.isdir(subfolder) and method_name[0] != "_":
                    method_details = {}
                    cur_json["detailed_results"][method_name] = method_details
                    for npy_filename in os.listdir(subfolder):
                        if npy_filename.endswith(".npy"):
                            method_details[npy_filename[:-4]] = np.load(
                                f"{subfolder}/{npy_filename}", allow_pickle=True
                            )
            cur_json["folder_path"] = folder
            result_jsons.append(cur_json)
            logger.debug(f"Loaded file {result_filename}")
        except Exception as e:
            logger.info(
                f"File {result_filename} cannot be read."
                f" Error {e.__class__.__name__} - {e}"
            )
    return result_jsons


def analyse_method(result_jsons, method_name, total_time, logger):
    jsons = list(filter(lambda json: method_name in json["accuracies"], result_jsons))
    if len(jsons) > 0:
        seeds = [json["seed"] for json in jsons]
        unique_seeds, unique_idc = np.unique(seeds, return_index=True)

        # Compute accuracies
        accuracies = np.array([json["accuracies"][method_name] for json in jsons])
        unique_acc = accuracies[unique_idc]

        error_perc = 100 * (1 - unique_acc)
        error_mean = np.mean(error_perc)
        error_std = np.std(error_perc)

        logger.info(
            f"Method {method_name:15} Seeds {unique_seeds}."
            f" Error rate: {error_mean:.4g} \pm {error_std:.3g}."
        )

        # Compute L1 marginals
        estimated_marginals = np.array(
            [
                json["detailed_results"][method_name]["est_label_marginals"]
                for json in jsons
                if method_name in json["detailed_results"]
            ]
        )
        true_marginals = np.array(
            [
                json["detailed_results"][method_name]["true_label_marginals"]
                for json in jsons
                if method_name in json["detailed_results"]
            ]
        )

        if len(estimated_marginals) > 0:
            l2_sqs = [
                lsu.get_l2_square_diff(
                    estimated_marginals[i, :, :], true_marginals[i, :, :]
                )
                / total_time
                for i in range(estimated_marginals.shape[0])
            ]

            logger.info(
                f"Method {method_name:15} Seeds {unique_seeds}."
                f" Marginal estimates L2 square:"
                f" {np.mean(l2_sqs):.4g} \pm {np.std(l2_sqs):.3g}."
            )
    else:
        logger.info(f"Method {method_name} has no corresponding experiments")


def analyse_unsupervised_results(
    data_names,
    shift_names,
    num_online_samples,
    total_time,
    source_test_ratio,
    use_source_test,
    calibration_type,
    shift_estimator_type,
    use_current_marginal_estimate,
    method_names,
    logger,
):
    # Loop over data, shift
    for data_name in data_names:
        for shift_name in shift_names:
            model_name = DATA_MODELS[data_name]
            logger.info("\n" + "*" * 84)
            logger.info(f"Data: {data_name}. Model: {model_name}. Shift: {shift_name}")

            # Find matching marg1, marg2, num_online_samples, total_time, calibration
            marg1, marg2 = DATA_MARGINALS[data_name]
            params = {
                "data_name": data_name,
                "base_model_name": model_name,
                "num_online_samples": num_online_samples,
                "total_time": total_time,
                "use_source_test": use_source_test,
                "source_test_ratio": source_test_ratio,
                "calibration_type": calibration_type,
                "shift_estimator_type": shift_estimator_type,
                "use_current_marginal_estimate": use_current_marginal_estimate,
                "shift_type": shift_name,
                "marg1": marg1,
                "marg2": marg2,
            }
            result_folder = slu.get_result_folder(
                params=params,
                root_path=init_path.parent_path,
                get_parent=True,
                mkdir=False,
            )
            result_jsons = collect_results(result_folder, logger)
            for method_name in method_names:
                analyse_method(result_jsons, method_name, total_time, logger)


if __name__ == "__main__":
    args = get_parser().parse_args()
    logger = mscu.get_logger(args.log_level)
    logger.info(f"Script parameters: {vars(args)}")

    # Loop over folders. Get valid results.json
    analyse_unsupervised_results(
        data_names=args.data_names.split(" "),
        shift_names=args.shift_names.split(" "),
        num_online_samples=args.num_online_samples,
        total_time=args.total_time,
        source_test_ratio=args.source_test_ratio,
        use_source_test=args.use_source_test,
        calibration_type=args.calibration_type,
        shift_estimator_type=args.shift_estimator_type,
        use_current_marginal_estimate=args.use_current_marginal_estimate,
        method_names=args.method_names.split(" "),
        logger=logger,
    )
