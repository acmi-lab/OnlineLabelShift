"""This scripts plots how different OLS methods perform over time
"""

from collections import defaultdict

import init_path
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

import utils.misc_utils as mscu
import utils.proj_utils as prju
import utils.saveload_utils as slu
from utils.data_utils import DATA_NUM_LABELS
from pathlib import Path
import utils.misc_utils as mscu
import json

DEFAULT_METHODS = [
    # "RW_DELAYED_BBSE",
    # "RW_FTFWH_BBSE",
    "UOGD_BBSE",
    # "ATLAS_BBSE",
    "RW_ROGD_BBSE",
    "RW_FTH_BBSE",
    "RW_FLH-FTL_BBSE",
]

params = {
    "data_name": "shl",
    "base_model_name": "mlp",
    "shift_type": "monotone",
    "use_current_marginal_estimate": False,
    "use_source_test": True,
    # "source_test_ratio": 0.02,
    "source_test_ratio": 1.0,
    "total_time": 7000,
    "calibration_type": "ts",
    "shift_estimator_type": "bbse",
    "num_online_samples": 10,
}


def get_default_shift_parameters(num_labels):
    marg1 = np.ones(num_labels) / num_labels
    marg2 = np.zeros(num_labels)
    marg2[0] = 1.0

    return marg1, marg2


marg1, marg2 = get_default_shift_parameters(DATA_NUM_LABELS[params["data_name"]])
params.update({"marg1": marg1, "marg2": marg2})

# Collect results
logger = mscu.get_logger(level="INFO")
result_folder = slu.get_result_folder(
    params, root_path=init_path.parent_path, mkdir=False, get_parent=True
)
results = slu.collect_results(result_folder, logger)

# Map method to list of accuracies, in order of time
methods_acc_avg, methods_acc_std = defaultdict(list), defaultdict(list)
methods_msq_avg, methods_msq_std = defaultdict(list), defaultdict(list)
for method in DEFAULT_METHODS:
    results_with_method = slu.filter_by_method(results, method)
    results_with_method, unique_seeds = slu.filter_by_seed(results_with_method)

    # Collect accuracies
    accs_over_time = np.array(
        [
            json["detailed_results"][method]["accuracies_per_time"]
            for json in results_with_method
        ]
    )
    methods_acc_avg[method] = np.mean(accs_over_time, axis=0)
    methods_acc_std[method] = np.std(accs_over_time, axis=0)

    # Collect msq
    if method not in ("base", "UOGD_BBSE", "ATLAS_BBSE"):
        # (num_seed x time x n_classes)
        tru_marg_over_time = np.array(
            [
                json["detailed_results"][method]["true_label_marginals"]
                for json in results_with_method
            ]
        )
        est_marg_over_time = np.array(
            [
                json["detailed_results"][method]["est_label_marginals"]
                for json in results_with_method
            ]
        )
        marg_sqerr = np.sum((tru_marg_over_time - est_marg_over_time) ** 2, axis=2)
        msq_avg = np.mean(marg_sqerr, axis=0)
        msq_std = np.std(marg_sqerr, axis=0)
        methods_msq_avg[method] = msq_avg
        methods_msq_std[method] = msq_std
print(f"{methods_acc_avg=}")
print(f"{methods_msq_avg=}")
# Plt
# plt.rcParams['text.usetex'] = True #Let TeX do the typsetting
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}', r'\sansmath'] #Force sans-serif math mode (for axes labels)
# plt.rcParams['font.family'] = 'sans-serif' # ... for regular text
# plt.rcParams['font.sans-serif'] = 'Helvetica, Avant Garde, Computer Modern Sans serif' # Choose a nice font here

fig, ax1 = plt.subplots()
plt.grid()
ax2 = ax1.twinx()

display_names = {
    "RW_DELAYED_BBSE": "BBSE",
    "UOGD_BBSE": "UOGD",
    "RW_ROGD_BBSE": "ROGD",
    "RW_FTH_BBSE": "FTH",
    "RW_FTFWH_BBSE": "FTFWH",
    "RW_FLH-FTL_BBSE": "FLH-FTL (ours)",
    "ATLAS_BBSE": "ATLAS",
}
colors = {
    "RW_DELAYED_BBSE": "olive",
    "UOGD_BBSE": "b",
    "RW_ROGD_BBSE": "m",
    "RW_FTH_BBSE": "k",
    "RW_FTFWH_BBSE": "c",
    "RW_FLH-FTL_BBSE": "r",
    "ATLAS_BBSE": "g",
}
for method_name in methods_acc_avg:
    logger.info(f"Method {method_name} Accs: {methods_acc_avg[method_name]}")
    # ax1.errorbar(
    #     x=np.arange(total_time),
    #     y=1 - methods_acc_avg[method_name],
    #     yerr=methods_acc_std[method_name],
    #     label=display_names[method_name],
    #     c=colors[method_name],
    # )
    smooth_accs = savgol_filter(1 - methods_acc_avg[method_name], 500, 1)
    ax1.plot(
        np.arange(params["total_time"]),
        smooth_accs,
        label=display_names[method_name],
        c=colors[method_name],
    )
    if method_name in methods_msq_avg:
        # ax2.errorbar(
        #     x=np.arange(total_time),
        #     y=methods_msq_avg[method_name],
        #     yerr=methods_msq_std[method_name],
        #     label=display_names[method_name],
        #     c=colors[method_name],
        #     linestyle="-."
        # )
        smooth_msqs = savgol_filter(methods_msq_avg[method_name], 500, 1)
        ax2.plot(
            np.arange(params["total_time"]),
            smooth_msqs,
            label=display_names[method_name],
            c=colors[method_name],
            linestyle="-.",
        )
plt.axvline(x=1.0, linestyle=":", linewidth=1.0)

plt.xlabel("Time")
fs = 16
ax2.set_ylabel("Mean Square Error", fontsize=fs)
# ax2.legend(fontsize=15)
# ax2.set_xticklabels([0.1, 0.15, 0.2], fontsize=fs)
ax2.tick_params(labelsize=fs)

ax1.set_xlabel("Time", fontsize=fs)
# plt.xticks([0.5, 1, 10, 50])
# ax1.set_xticklabels([0.5, 1, 10, 50])
ax1.set_ylabel("Classification Error", fontsize=fs)
ax1.legend(fontsize=12, ncol=3)
ax1.tick_params(labelsize=fs)

folder = prju.Parameters.get_plot_path(root_path=init_path.parent_path)
folder = (
    Path(folder)
    / "acc-over-time"
    / (
        f"d={params['data_name']}_st={params['source_test_ratio']}"
        f"_usecur={params['use_current_marginal_estimate']}"
    )
)
folder.mkdir(parents=True, exist_ok=True)
plt.savefig(str(folder) + "/acc-marg.pdf", transparent=True, bbox_inches="tight")

with open(str(folder) + f"/param.json", mode="w") as f:
    json.dump(mscu.serialize_dict(params), f, indent=2)
