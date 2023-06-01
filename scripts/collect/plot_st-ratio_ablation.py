"""This scripts plots how different OLS methods perform under changing stest ratio

1. Make x axis * 50
2. plt.axvline at 1
3. plt.xlabel("Fraction")
3.a ax1.ylabel("Classification Error")
3.b ax1.ylabel("Mean Square Error")
4. standard deviation
"""

from collections import defaultdict

import init_path
import matplotlib.pyplot as plt
import numpy as np

import utils.misc_utils as mscu
import utils.proj_utils as prju
import utils.saveload_utils as slu
from utils.data_utils import DATA_NUM_LABELS
from pathlib import Path
import utils.misc_utils as mscu
import json

METHODS_TO_PLOT = [
    # "base",
    "RW_ROGD_BBSE",
    "UOGD_BBSE",
    # "ATLAS_BBSE",
    "RW_FTH_BBSE",
    # "RW_FTFWH_BBSE",
    "RW_FLH-FTL_BBSE",
]

data_name = "cifar10"
model_name = "resnet18"
shift_type = "bernouli"
use_source_test = True
total_time = 1000
calibration_type = "ts"
shift_estimator_type = "bbse"
use_current_marginal_estimate = False
num_samples = 10
source_test_ratios = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0]

logger = mscu.get_logger(level="INFO")


def get_default_shift_parameters(num_labels):
    marg1 = np.ones(num_labels) / num_labels
    marg2 = np.zeros(num_labels)
    marg2[0] = 1.0

    return marg1, marg2


marg1, marg2 = get_default_shift_parameters(DATA_NUM_LABELS[data_name])

# Collects method accuracies
methods_acc_avg, methods_acc_std = defaultdict(list), defaultdict(list)
methods_msq_avg, methods_msq_std = defaultdict(list), defaultdict(list)

for source_test_ratio in source_test_ratios:
    params = {
        "data_name": data_name,
        "base_model_name": model_name,
        "shift_type": shift_type,
        "shift_estimator_type": shift_estimator_type,
        "num_online_samples": num_samples,
        "total_time": total_time,
        "use_current_marginal_estimate": use_current_marginal_estimate,
        "use_source_test": use_source_test,
        "source_test_ratio": source_test_ratio,
        "calibration_type": calibration_type,
        "marg1": marg1,
        "marg2": marg2,
    }
    result_folder = slu.get_result_folder(
        params, root_path=init_path.parent_path, mkdir=False, get_parent=True
    )
    results = slu.collect_results(result_folder, logger)

    for method in METHODS_TO_PLOT:
        results_with_method = slu.filter_by_method(results, method)
        results_with_method, unique_seeds = slu.filter_by_seed(results_with_method)
        if len(unique_seeds) != 3:
            print(f"Unexpected number of seeds for {method}. Seeds: {unique_seeds}")

        all_accs = [
            result["accuracies"][method]
            for result in results_with_method
            if method in result["accuracies"]
        ]
        methods_acc_avg[method].append(np.mean(all_accs))
        methods_acc_std[method].append(np.std(all_accs))

        if method not in ("base", "UOGD_BBSE", "ATLAS_BBSE"):
            uniq_tru_marginals = [
                json["detailed_results"][method]["true_label_marginals"]
                for json in results_with_method
            ]
            uniq_est_marginals = [
                json["detailed_results"][method]["est_label_marginals"]
                for json in results_with_method
            ]
            uniq_msq = [
                np.sum((uniq_tru_marginals[i] - uniq_est_marginals[i]) ** 2) / total_time
                for i in range(len(uniq_est_marginals))
            ]
            methods_msq_avg[method].append(np.mean(uniq_msq))
            methods_msq_std[method].append(np.std(uniq_msq))


# Plotting parameters
display_names = {
    "base": "Base",
    "RW_DELAYED_BBSE": "BBSE",
    "UOGD_BBSE": "UOGD",
    "RW_ROGD_BBSE": "ROGD",
    "RW_FTH_BBSE": "FTH",
    "RW_FTFWH_BBSE": "FTFWH",
    "RW_FLH-FTL_BBSE": "FLH-FTL (ours)",
    "ATLAS_BBSE": "ATLAS",
}
colors = {
    "base": "purple",
    "RW_DELAYED_BBSE": "olive",
    "UOGD_BBSE": "b",
    "RW_ROGD_BBSE": "m",
    "RW_FTH_BBSE": "k",
    "RW_FTFWH_BBSE": "c",
    "RW_FLH-FTL_BBSE": "r",
    "ATLAS_BBSE": "g",
}

# Plot
fig, ax1 = plt.subplots()
plt.grid()
ax2 = ax1.twinx()

source_test_ratios = np.array(source_test_ratios) * 50
for method_name in methods_acc_avg:
    logger.info(f"Method {method_name} Accs: {methods_acc_avg[method_name]}")
    ax1.errorbar(
        x=source_test_ratios,
        y=1 - np.array(methods_acc_avg[method_name]),
        yerr=methods_acc_std[method_name],
        label=display_names[method_name],
        c=colors[method_name],
    )
    if method_name in methods_msq_avg:
        ax2.errorbar(
            x=source_test_ratios,
            y=methods_msq_avg[method_name],
            yerr=methods_msq_std[method_name],
            label=display_names[method_name],
            c=colors[method_name],
            linestyle="-.",
        )
plt.axvline(x=1.0, linestyle=":", linewidth=1.0)


plt.xscale("log")
plt.xlabel("Fraction")
fs = 17
ax2.set_ylabel("Mean Square Error", fontsize=fs)
# ax2.legend(fontsize=15)
ax2.tick_params(labelsize=fs)

ax1.set_xlabel("Fraction", fontsize=fs)
plt.xticks([0.5, 1, 10, 50])
ax1.set_xticklabels([0.5, 1, 10, 50], fontsize=fs)
ax1.set_ylabel("Classification Error", fontsize=fs)
ax1.legend(fontsize=12, ncol=3)
ax1.tick_params(labelsize=fs)

folder = prju.Parameters.get_plot_path(root_path=init_path.parent_path)
folder = (
    Path(folder)
    / "st-ratio_ablation"
    / (
        f"d={params['data_name']}_st={params['source_test_ratio']}"
        f"_usecur={params['use_current_marginal_estimate']}"
    )
)
folder.mkdir(parents=True, exist_ok=True)

plt.savefig(str(folder) + "/st-ratio.pdf", transparent=True, bbox_inches="tight")
with open(str(folder) + f"/param.json", mode="w") as f:
    json.dump(mscu.serialize_dict(params), f, indent=2)
