"""This scripts plots how different OLS methods are tracking the true label marginals
"""
import argparse
from pathlib import Path

import init_path
import matplotlib.pyplot as plt
import numpy as np
from collect_ols_results import (
    DATA_MARGINALS,
    DATA_MODELS,
    DEFAULT_METHODS,
    collect_results,
)

import utils.misc_utils as mscu
import utils.proj_utils as prju

DEFAULT_METHODS = [
    "RW_DELAYED_BBSE",
    # "UOGD_BBSE",
    "RW_ROGD_BBSE",
    "RW_FTH_BBSE",
    "RW_FTFWH_BBSE",
    "RW_FLH-FTL_BBSE",
]

MARG_KEY = "detailed_results"
TRU_KEY = "true_label_marginals"
EST_KEY = "est_label_marginals"


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data-name", type=str, required=False, default="synthetic"
    )
    parser.add_argument(
        "-s", "--shift-name", type=str, required=False, default="monotone"
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
    parser.add_argument("--seed", default=42, type=int, required=False)
    parser.add_argument("-l", "--log-level", default="INFO", required=False)
    parser.add_argument("--source-test-ratio", default=1.0, type=float, required=False)
    parser.add_argument(
        "--use-source-test", default=True, type=mscu.boolean, required=False
    )

    return parser

def filter_results_by_parameters(
    result_jsons,
    data_name,
    shift_name,
    num_online_samples,
    total_time,
    seed,
    source_test_ratio,
    use_source_test,
    logger,
):
    marg1, marg2 = DATA_MARGINALS[data_name]
    logger.info(
        f"Filtering for data_name={data_name}"
        f" base_model_name={DATA_MODELS[data_name]}"
        f" shift_type={shift_name}"
        f" num_online_samples={num_online_samples}"
        f" total_time={total_time}"
        f" source_test_ratio={source_test_ratio}"
        f" use_source_test={use_source_test}"
        f" marg1={marg1}"
        f" marg2={marg2}"
    )
    filt_jsons = filter(
        lambda json: (
            json["data_name"] == data_name
            and json["base_model_name"] == DATA_MODELS[data_name]
            and json["shift_type"] == shift_name
            and json["num_online_samples"] == num_online_samples
            and json["total_time"] == total_time
            and json["seed"] == seed
            and json["source_test_ratio"] == source_test_ratio
            and json["use_source_test"] == use_source_test
            and json["marg1"] == str(marg1)
            and json["marg2"] == str(marg2)
        ),
        result_jsons,
    )
    return list(filt_jsons)


def plot_all_methods(relevant_results, data_name, method_names, total_time, logger):
    if len(relevant_results) == 0:
        logger.info("No results found")
        return

    # Collect method names from each json
    experiment_ids = [Path(json["folder_path"]).stem for json in relevant_results]
    logger.info(f"Combining results from {experiment_ids}")
    combined_json = {**relevant_results[0]}
    for result_json in relevant_results[1:]:
        for method_name in result_json[MARG_KEY]:
            if method_name in combined_json[MARG_KEY]:
                if not np.all(
                    combined_json[MARG_KEY][method_name][TRU_KEY]
                    == result_json[MARG_KEY][method_name][TRU_KEY]
                ):
                    print(f"{method_name=}")
            else:
                combined_json[MARG_KEY][method_name] = result_json[MARG_KEY][method_name]

    # Create plot folders for all experiments meeting the specifications
    plot_folders = [
        Path(json["folder_path"]) / "_marginal_plots" for json in relevant_results
    ]
    for folder in plot_folders:
        folder.mkdir(parents=True, exist_ok=True)

    data_name = combined_json["data_name"]
    shift_name = combined_json["shift_type"]

    # Plot marginals for each label
    dataset_params = prju.DatasetParameters.dataset_defaults[data_name]
    for label in range(dataset_params["num_classes"]):
        plt.title(f"Data {data_name} Shift {shift_name} Label {label}")

        for idx, method_name in enumerate(method_names.split(" ")):
            if idx == 0:
                tru_marg = combined_json[MARG_KEY][method_name][TRU_KEY][:, label]
                plt.plot(range(total_time), tru_marg, label="True marginal")

            est_marg = combined_json[MARG_KEY][method_name][EST_KEY][:, label]
            plt.plot(
                range(total_time),
                est_marg,
                label=f"{method_name} estimated marginal",
            )

        # Save
        plt.legend()
        for folder in plot_folders:
            plt.savefig(folder / f"label={label}.jpg")
        plt.clf()

    for folder in plot_folders:
        logger.info(f"Saved plots to {folder}")


def plot_individaul_methods(relevant_results, data_name, total_time):
    dataset_params = prju.DatasetParameters.dataset_defaults[data_name]
    for result in relevant_results:
        for method_name in result[MARG_KEY]:
            plot_folder = Path(result["folder_path"]) / method_name / "_marginal_plots"
            plot_folder.mkdir(parents=True, exist_ok=True)

            for label in range(dataset_params["num_classes"]):
                plt.title(f"Label {label}")

                # Plot
                tru_marg = result[MARG_KEY][method_name][TRU_KEY][:, label]
                est_marg = result[MARG_KEY][method_name][EST_KEY][:, label]
                plt.plot(range(total_time), tru_marg, label="True marginal")
                plt.plot(range(total_time), est_marg, label="Estimated marginal")
                plt.legend()
                plt.savefig(plot_folder / f"label={label}.jpg")
                plt.clf()


def plot_marginal_estimates(
    result_jsons,
    data_name,
    shift_name,
    num_online_samples,
    method_names,
    total_time,
    seed,
    source_test_ratio,
    use_source_test,
    logger,
):
    relevant_results = filter_results_by_parameters(
        result_jsons,
        data_name,
        shift_name,
        num_online_samples,
        total_time,
        seed,
        source_test_ratio,
        use_source_test,
        logger,
    )
    logger.info(f"Plotting all methods in {method_names}")
    plot_all_methods(relevant_results, data_name, method_names, total_time, logger)
    logger.info("Done")

    experiments = [Path(result["folder_path"]).stem for result in relevant_results]
    logger.info(f"Plotting individual methods in experiments {experiments}")
    plot_individaul_methods(relevant_results, data_name, total_time)
    logger.info("Done")


if __name__ == "__main__":
    args = get_parser().parse_args()
    logger = mscu.get_logger(args.log_level)
    logger.info(f"Script parameters: {vars(args)}")

    # Loop over folders. Get valid results.json
    result_jsons = collect_results(logger=logger)
    plot_marginal_estimates(
        result_jsons=result_jsons,
        data_name=args.data_name,
        shift_name=args.shift_name,
        method_names=args.method_names,
        num_online_samples=args.num_online_samples,
        total_time=args.total_time,
        seed=args.seed,
        source_test_ratio=args.source_test_ratio,
        use_source_test=args.use_source_test,
        logger=logger,
    )
