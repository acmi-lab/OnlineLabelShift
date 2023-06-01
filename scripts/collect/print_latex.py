import itertools

import init_path
import numpy as np

import utils.label_shift_utils as lsu
import utils.misc_utils as mscu
import utils.saveload_utils as slu
from utils.data_utils import DATA_NUM_LABELS

MODELS = {
    "synthetic": "logreg",
    # "mnist": "fcn_early",
    "mnist": "randforest",
    # "cifar10": "resnet18",
    "cifar10": "randforest",
    "eurosat": "resnet18",
    "fashion": "mlp",
    "shl": "mlp",
    "arxiv": "bert",
}

fixed_params = {
    # "data_name": "synthetic",
    # "data_name": "mnist",
    # "data_name": "cifar10",
    "source_test_ratio": 0.02,
    # "source_test_ratio": 1.0,
    "calibration_type": "ts",
    "use_current_marginal_estimate": False,
    # "use_current_marginal_estimate": True,
    "num_online_samples": 10,
    "shift_estimator_type": "bbse",
    # "total_time": 7000,
    "total_time": 1000,
    "use_source_test": True,
    "shift_type": "bernouli",
    # "shift_type": "monotone",
}

row_param_dict = {
    # "shift_type": ["monotone", "bernouli"],
    # "shift_type": ["sinusoidal", "square"],
    # "shift_type": ["bernouli", "sinusoidal"],
    # "data_name": ["shl"],
    "data_name": ["cifar10"],
    # "data_name": ["mnist"],
    # "data_name": ["mnist", "cifar10"],
    # "method": [
    #     "base",
    #     "ofc",
    #     # "RW_ORACLE",
    #     "RW_FTH",
    #     "RW_FTFWH",
    #     "RW_ROGD",
    #     "UOGD",
    #     "ATLAS",
    #     "RW_FLH-FTL",
    #     # "RW_SIMP-LOCAL",
    #     # "LIN-RETRAIN_FLH-FTL",
    # ],
}


col_param_dict = {
    # "data_name": ["synthetic", "mnist", "cifar10", "eurosat", "fashion", "arxiv"],
    # "data_name": ["cifar10"],
    # "data_name": ["synthetic"],
    # "data_name": ["synthetic", "cifar10", "fashion"],
    # "data_name": ["synthetic", "cifar10"],
    # "shift_estimator_type": ["bbse", "mlls", "rlls"],
    # "source_test_ratio": [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0],
    # "shift_type": ["bernouli", "sinusoidal"],
    # "shift_type": ["monotone", "square"],
    # "shift_type": ["monotone", "bernouli"],
    # "num_online_samples": [1, 2, 4, 8, 16, 32, 64, 128],
    "method": [
        "base",
        # "ofc",
        "RW_ORACLE",
        "RW_ROGD",
        "RW_FTH",
        "RW_FTFWH",
        # "RW_ROGD",
        # "UOGD",
        # "ATLAS",
        "RW_FLH-FTL",
        # "RW_SIMP-LOCAL",
    ],
}


def get_default_shift_parameters(num_labels):
    marg1 = np.ones(num_labels) / num_labels
    marg2 = np.zeros(num_labels)
    # marg2[min(5, num_labels - 1)] = 1.0
    # marg2[-1] = 1.0
    marg2[0] = 1.0

    return marg1, marg2


DATA_MARGINALS = {
    data_name: get_default_shift_parameters(DATA_NUM_LABELS[data_name])
    for data_name in DATA_NUM_LABELS
}


def list_params_comb(param_dict):
    """Turn row/col params into list of combinations
    For example, [..., {"shift_type": ?, "method": base}]
    """
    param_types = list(param_dict.keys())
    combs = itertools.product(*[param_dict[param_type] for param_type in param_types])

    params = []
    for comb in combs:
        cur_params = {}
        for idx, param_type in enumerate(param_types):
            cur_params[param_type] = comb[idx]
        params.append(cur_params)
    return params


print(f"Fixed params: {fixed_params}")

# Display columns
for col_params in list_params_comb(col_param_dict):
    print(f"{tuple(col_params.values())} ", end="")
print()

# Collect results
for row_params in list_params_comb(row_param_dict):
    latex_str = ""
    latex_marg_str = ""
    for col_params in list_params_comb(col_param_dict):
        params = {**fixed_params, **col_params, **row_params}
        params["base_model_name"] = MODELS[params["data_name"]]
        params["marg1"], params["marg2"] = DATA_MARGINALS[params["data_name"]]

        folder = slu.get_result_folder(
            params, root_path=init_path.parent_path, get_parent=True, mkdir=False
        )
        results = slu.collect_results(folder, logger=mscu.get_logger())

        # Extract method name
        method = params["method"]
        if method not in ["base", "ofc"]:
            method += f"_{params['shift_estimator_type'].upper()}"

        jsons_with_method = slu.filter_by_method(results, method)

        # Filter by unique seeds
        uniq_results, uniq_seeds = slu.filter_by_seed(jsons_with_method)
        if len(uniq_results) != 3:
            print(
                f"Warning: unexpected number of results."
                f" Folder {folder} has {len(uniq_results)} results of {method} with seeds {uniq_seeds}"
            )

        # Get mean and error
        errors = 100 * np.array([1 - json["accuracies"][method] for json in uniq_results])
        mean = np.mean(errors)
        std = np.std(errors)

        # Formatting
        mean_sigf = f"{mean:.2g}"
        period_idx = mean_sigf.find(".")

        # Set same sig fig for mean and error
        decimals = 0 if period_idx == -1 else len(mean_sigf[period_idx + 1 :])
        rmean, rstd = np.around(mean, decimals), np.around(std, decimals)

        # Remove trailing zeros
        if str(rmean).find(".") == 2:
            rmean, rstd = rmean.astype(int), rstd.astype(int)

        latex_str += f"\eentry{{{rmean}}}{{{rstd}}} & "

        # Get marginal data
        estimated_marginals = np.array(
            [
                json["detailed_results"][method]["est_label_marginals"]
                for json in uniq_results
                if method in json["detailed_results"]
            ]
        )
        true_marginals = np.array(
            [
                json["detailed_results"][method]["true_label_marginals"]
                for json in uniq_results
                if method in json["detailed_results"]
            ]
        )

        if len(estimated_marginals) > 0:
            l2_sqs = [
                lsu.get_l2_square_diff(
                    estimated_marginals[i, :, :], true_marginals[i, :, :]
                )
                / params["total_time"]
                for i in range(estimated_marginals.shape[0])
            ]

            mean, std = np.mean(l2_sqs), np.std(l2_sqs)

            # Add trailing zeros
            latex_marg_str += f"\eentry{{{mean:.2f}}}{{{std:.2f}}} & "
        else:
            latex_marg_str += f"\\nentry & "

    latex_str = latex_str[:-2]
    latex_str += "\\\\"

    latex_marg_str = latex_marg_str[:-2]
    latex_marg_str += "\\\\"

    print(f"{str(row_params)}:\n{latex_str}")
    if len(latex_marg_str) > 0:
        print(f"Marg :\n{latex_marg_str}")
    print()
