from pathlib import Path

import numpy as np

import utils.misc_utils as mscu
import utils.proj_utils as prju
import os
import json

def get_result_folder(params, root_path, get_parent=False, mkdir=True):
    folder = Path(prju.Parameters.get_result_path(root_path=root_path))
    cur_time = mscu.get_current_time()

    # Construct parameter-based folder path
    layer1 = (
        f"d={params['data_name']}_m={params['base_model_name']}"
        f"_shift-estimator-type={params['shift_estimator_type']}"
    )
    layer2 = (
        f"num-samples={params['num_online_samples']}_t={params['total_time']}"
        f"_use_current={params['use_current_marginal_estimate']}"
    )
    layer3 = (
        f"use_source_test={params['use_source_test']}"
        f"_st-ratio={params['source_test_ratio']}"
        f"_cal-type={params['calibration_type']}"
    )
    marg1, marg2 = np.round(params["marg1"], 2), np.round(params["marg2"], 2)
    layer4 = f"s={params['shift_type']}_m1={marg1}_m2={marg2}"
    folder = folder / layer1 / layer2 / layer3 / layer4

    if not get_parent:
        layer5 = f"seed={params['seed']}_time={cur_time}"
        folder = folder / layer5

    if mkdir:
        folder.mkdir(parents=True, exist_ok=False)
    return folder

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

def filter_by_seed(results):
    seeds = [result["seed"] for result in results]
    unique_seeds, unique_idc = np.unique(seeds, return_index=True)

    unique_results = [results[unique_idx] for unique_idx in unique_idc]
    return unique_results, unique_seeds

def filter_by_method(results, method):
    return [result for result in results if method in result["accuracies"]]
