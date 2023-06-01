"""This script runs the specified online label shift experiment
"""
import argparse
import copy
import json
from logging import FileHandler
from pathlib import Path

import init_path
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

import utils.data_utils as datu
import utils.label_shift_utils as lsu
import utils.misc_utils as mscu
import utils.model_utils as modu
import utils.proj_utils as prju
import utils.saveload_utils as slu
from utils.model_utils import SUPPORTED_LOCAL_SHIFT_ESTIMATORS

SUPPORTED_SHIFTS = [
    "monotone",
    "square",
    "sinusoidal",
    "bernouli",
]


def save_ols_results(results, folderpath, model_name):
    folder = Path(folderpath) / model_name
    folder.mkdir(parents=True, exist_ok=False)
    for result_name in results:
        np.save(folder / f"{result_name}.npy", results[result_name])


def get_default_shift_parameters(num_labels):
    marg1 = np.ones(num_labels) / num_labels
    marg2 = np.zeros(num_labels)
    marg2[0] = 1.0

    return marg1, marg2


def set_marginals(args):
    # Set default shift parameters
    num_labels = datu.DATA_NUM_LABELS[args.data_name]
    default_m1, default_m2 = get_default_shift_parameters(num_labels)

    if args.marg1 is None:
        args.marg1 = default_m1
    if args.marg2 is None:
        args.marg2 = default_m2

    if args.marg1.shape[0] != num_labels or args.marg2.shape[0] != num_labels:
        raise AssertionError(
            f"Shfit dimension mismatch. {args.marg1.shape[0]=}"
            f" {args.marg2.shape[0]=} {num_labels=}"
        )


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data-name",
        default="synthetic",
        type=str,
        required=False,
        choices=datu.DATA_NUM_LABELS.keys(),
    )
    parser.add_argument(
        "-m", "--base-model-name", default="logreg", type=str, required=False
    )
    parser.add_argument(
        "-s",
        "--shift-type",
        default="monotone",
        type=str,
        required=False,
        choices=SUPPORTED_SHIFTS,
    )
    parser.add_argument("--marg1", type=str, required=False, default=None)
    parser.add_argument("--marg2", type=str, required=False, default=None)
    parser.add_argument(
        "-n", "--num-online-samples", default=10, type=int, required=False
    )
    parser.add_argument("-b", "--batch-size", default=200, type=int, required=False)
    parser.add_argument("-t", "--total-time", default=500, type=int, required=False)
    parser.add_argument("--seed", default=8610, type=int, required=False)
    parser.add_argument("-l", "--log-level", default="INFO", required=False)
    parser.add_argument("--source-test-ratio", default=1.0, type=float, required=False)
    parser.add_argument(
        "--use-source-test",
        default=True,
        type=mscu.boolean,
        required=False,
        help=(
            "Determines whether all online classifiers get to use source test data "
            " aside from confusion matrix"
        ),
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
    parser.add_argument("--do-ofc", default=False, type=mscu.boolean, required=False)
    parser.add_argument("--ofc-ntries", default=10, type=int, required=False)
    parser.add_argument("--do-rogd", default=False, type=mscu.boolean, required=False)
    parser.add_argument(
        "--use-default-rogd-lipschitz", default=True, type=mscu.boolean, required=False
    )
    parser.add_argument(
        "--rogd-use-smooth", default=True, type=mscu.boolean, required=False
    )
    parser.add_argument("--rogd-lrs", default=None, type=str, required=False)
    parser.add_argument("--do-fth", default=False, type=mscu.boolean, required=False)
    parser.add_argument("--do-ftfwh", default=False, type=mscu.boolean, required=False)
    parser.add_argument("--do-uogd", default=False, type=mscu.boolean, required=False)
    parser.add_argument("--uogd-lrs", default=None, type=str, required=False)
    parser.add_argument("--do-atlas", default=False, type=mscu.boolean, required=False)
    parser.add_argument("--do-flhftl", default=False, type=mscu.boolean, required=False)
    parser.add_argument(
        "--do-simp-local", default=False, type=mscu.boolean, required=False
    )
    parser.add_argument(
        "--do-marg-oracle", default=False, type=mscu.boolean, required=False
    )
    parser.add_argument("--do-retrain", default=False, type=mscu.boolean, required=False)
    parser.add_argument("--retrain-lr", default=0.5, type=float, required=False)
    parser.add_argument("--retrain-epoch", default=50, type=int, required=False)
    parser.add_argument("--do-bnadapt", default=False, type=mscu.boolean, required=False)
    parser.add_argument("--do-all", default=False, type=mscu.boolean, required=False)
    parser.add_argument(
        "--calibration-type", default="ts", type=str, required=False, choices=["ts", "vs"]
    )
    parser.add_argument("--save", default=False, type=mscu.boolean, required=False)

    return parser


def run_online_label_shift_experiment(
    data_name,
    base_model_name,
    shift_type,
    marg1,
    marg2,
    num_online_samples,
    batch_size,
    total_time,
    logger,
    seed,
    source_test_ratio,
    use_source_test,
    do_ofc,
    ofc_ntries,
    do_rogd,
    use_default_rogd_lipschitz,
    rogd_use_smooth,
    rogd_lrs,
    do_fth,
    do_ftfwh,
    do_uogd,
    uogd_lrs,
    do_atlas,
    do_flhftl,
    do_simp_local,
    do_marg_oracle,
    do_retrain,
    retrain_lr,
    retrain_epoch,
    do_bnadapt,
    do_all,
    calibration_type,
    shift_estimator_type,
    use_current_marginal_estimate,
):
    # Get model
    model_cls = modu.get_base_model_class(data_name, base_model_name)
    uncalibrated_model = model_cls(pretrained=True, root_path=init_path.parent_path)

    # Get data
    datasets = datu.get_datasets(
        data_name, source=True, target=True, root_path=init_path.parent_path
    )
    num_labels = datu.DATA_NUM_LABELS[data_name]

    source_train_dataset = datasets["source_train"]
    source_test_dataset = datasets["source_test"]
    source_test_dataset, _ = datu.split_dataset(
        source_test_dataset, num_labels, frac=source_test_ratio, seed=seed
    )

    source_train_marginals = lsu.get_label_marginals(source_train_dataset.y_array)
    source_test_marginals = lsu.get_label_marginals(source_test_dataset.y_array)

    logger.info(f"Device used: {prju.Parameters.device}")
    logger.info(f"Number of source train data: {len(source_train_dataset)}")
    logger.info(f"Source train data label marginals: {source_train_marginals}")
    logger.info(f"Number of source test data: {len(source_test_dataset)}")
    logger.info(f"Source test data label marginals: {source_test_marginals}")

    # Prepare the reference data ROGD and UOGD will use for updates
    ref_dataset = source_test_dataset if use_source_test else source_train_dataset
    ref_dataloader = DataLoader(ref_dataset, batch_size=batch_size, shuffle=False)

    # Get calibrator if using source test data
    if use_source_test:
        logger.info("Preparing calibrator")
        calibrator = modu.get_calibrator(
            uncalibrated_model,
            ref_dataloader,
            device=prju.Parameters.device,
            calibration_type=calibration_type,
            print_verbose=True,
        )
        logger.info(f"{calibrator.__class__=}")
        calibrated_model = copy.deepcopy(uncalibrated_model)
        calibrated_model.calibrator = calibrator

        logger.info("Calibrator prepared")
    # if use_source_test, calibrate the base model
    base_model = calibrated_model if use_source_test else uncalibrated_model

    # Generate shfits
    if data_name != "shl":
        logger.info("Sampling label-shifted data")
        logger.info(f"Number of Target data: {len(datasets['target'])}")
        shifts = lsu.get_shifts(
            shift_type, marg1=marg1, marg2=marg2, total_time=total_time
        )
        shifted_dataloaders = lsu.get_shifted_dataloaders(
            datasets["target"],
            shifts,
            num_samples=num_online_samples,
            batch_size=batch_size,
            num_labels=num_labels,
            seed=seed,
        )
        logger.debug(f"Number of shifts {len(shifted_dataloaders)}")
        logger.info("Sampling done")
    else:  # SHL is natural shift. No need to generate
        shifted_dataloaders = [
            DataLoader(
                datu.Subset(
                    datasets["target"],
                    indices=np.arange(
                        num_online_samples * t, num_online_samples * (t + 1)
                    ),
                ),
                shuffle=False,
                batch_size=num_online_samples,
            )
            for t in range(total_time)
        ]
        logger.info(f"Using natural shift for {data_name}")

    # Get local shift estimator component
    logger.info(f"Using local estimator {shift_estimator_type.upper()}")
    source_test_dataloader = DataLoader(
        source_test_dataset, batch_size=batch_size, shuffle=True
    )
    if shift_estimator_type == "bbse":
        local_shift_estimator = modu.BlackBoxShiftEstimator(
            model=copy.deepcopy(base_model),
            source_train_marginals=source_train_marginals,
            num_labels=num_labels,
            ref_dataloader=source_test_dataloader,  # Always use source test for BBSE
        )
        logger.info(f"BBSE confusion matrix: {local_shift_estimator.confusion_matrix}")
    elif shift_estimator_type == "rlls":
        local_shift_estimator = modu.RegularizedShiftEstimator(
            model=copy.deepcopy(base_model),
            source_train_marginals=source_train_marginals,
            source_train_num_data=len(source_train_dataset),
            num_labels=num_labels,
            ref_dataloader=source_test_dataloader,  # Always use source test for RLLS
        )
        logger.info(f"RLLS confusion matrix: {local_shift_estimator.confusion_matrix}")
    elif shift_estimator_type == "mlls":
        source_test_prob = copy.deepcopy(base_model).get_predicted_probabilities(
            dataloader=source_test_dataloader
        )
        source_soft_marginals = torch.mean(source_test_prob, axis=0).cpu().numpy()
        local_shift_estimator = modu.MaximumLikelihoodShiftEstimator(
            model=copy.deepcopy(base_model),
            source_soft_marginals=source_soft_marginals,
            num_labels=num_labels,
        )
        # Temporary measure to fix MLLS and UOGD compatibility issue
        local_shift_estimator.confusion_matrix = (
            local_shift_estimator.model.get_soft_confusion_matrix(
                dataloader=source_test_dataloader, num_labels=num_labels
            )
        )

    # Collect results
    # Get base classifier error
    logger.info("Computing base accuracy")
    dataset = ConcatDataset([dataloader.dataset for dataloader in shifted_dataloaders])
    total_dataloader = DataLoader(dataset, batch_size=batch_size)
    y_pred, y_true = copy.deepcopy(base_model).get_predictions(
        total_dataloader, verbose=True
    )
    base_acc = lsu.get_accuracy(y_true, y_pred)
    logger.info(f"Total number of samples {len(dataset)}")
    logger.info(f"Base accuracy: {base_acc}")

    accuracies = {"base": base_acc.item()}
    results = {"accuracies": accuracies, "detailed_results": {}}
    if do_ofc or do_all:
        start_time = mscu.tick()
        logger.info("Computing OFC accuracy")

        # Get OFC error
        ofc = modu.OptimalFixedReweightClassifier(
            model=copy.deepcopy(base_model),
            source_train_marginals=source_train_marginals,
        )
        ofc_acc = ofc.get_accuracy(total_dataloader, verbose=True, ntries=ofc_ntries)
        accuracies["ofc"] = ofc_acc
        logger.info(f"Optimal fixed classifier's accuracy: {ofc_acc}")
        logger.info(f"OFC completed in {mscu.tock(start_time):.3g} mins")

    if (do_uogd or do_all) and isinstance(
        uncalibrated_model, modu.BaseModelWithLinearWeight
    ):
        lrs = [None] if uogd_lrs is None else [None, *[float(lr) for lr in uogd_lrs]]
        for lr in lrs:
            start_time = mscu.tick()
            logger.info("Computing UOGD accuracy")

            # Get UOGD classifier error
            uogd = modu.UnbiasedOnlineGradientDescentClassifier(
                model=copy.deepcopy(uncalibrated_model),
                ref_dataloader=ref_dataloader,
                num_labels=num_labels,
                marginal_estimator=local_shift_estimator,
                weight_name="linear.weight",
                source_train_marginals=source_train_marginals,
                use_current_marginal_estimate=use_current_marginal_estimate,
                lr=lr if lr is None else torch.tensor(lr, device=prju.Parameters.device),
                clf_name="UOGD" if lr is None else f"UOGD_lr={lr}",
            )
            if lr is None:
                uogd.compute_lr_from_total_time(total_time=total_time)
                logger.info(f"{uncalibrated_model.weight_lipschitz_estimate=}")
            result = uogd.predict_online_label_shift(shifted_dataloaders)

            # Saving results
            method_name = uogd.name
            accuracies[method_name] = result["accuracy"].item()
            results["detailed_results"][method_name] = result
            results["uogd-natural-lr"] = uogd.lr.item()

            logger.info(f"{method_name} classifier's accuracy: {result['accuracy']}")
            logger.info(f"{method_name} completed in {mscu.tock(start_time):.3g} mins")

    if (do_atlas or do_all) and isinstance(
        uncalibrated_model, modu.BaseModelWithLinearWeight
    ):
        start_time = mscu.tick()
        logger.info("Computing ATLAS accuracy")

        atlas = modu.ATLAS(
            model=copy.deepcopy(uncalibrated_model),
            ref_dataloader=ref_dataloader,
            num_labels=num_labels,
            marginal_estimator=local_shift_estimator,
            weight_name="linear.weight",
            total_time=total_time,
            source_train_marginals=source_train_marginals,
            use_current_marginal_estimate=use_current_marginal_estimate,
        )
        result = atlas.predict_online_label_shift(shifted_dataloaders)

        # Saving results
        method_name = atlas.name
        accuracies[method_name] = result["accuracy"].item()
        results["detailed_results"][method_name] = result
        results["atlas-base-lr-pool"] = [lr.item() for lr in atlas.base_lr_pool]
        results["atlas-meta-lr"] = atlas.meta_lr

        logger.info(f"{method_name} classifier's accuracy: {result['accuracy']}")
        logger.info(f"{method_name} completed in {mscu.tock(start_time):.3g} mins")

    if do_retrain and isinstance(uncalibrated_model, modu.BaseModelWithLinearWeight):
        start_time = mscu.tick()
        flhftl_estimator = modu.FollowLeadingHistoryFollowTheLeaderEstimator(
            source_train_marginals=source_train_marginals,
            underlying_estimator=local_shift_estimator,
            num_labels=num_labels,
        )
        source_test_dataloader = DataLoader(
            source_test_dataset, batch_size=batch_size, shuffle=False
        )

        clf = modu.LinearLayerRetrainedClassifier(
            model=copy.deepcopy(uncalibrated_model),
            ref_dataloader=source_test_dataloader,
            marginal_estimator=flhftl_estimator,
            num_labels=num_labels,
            source_train_marginals=source_train_marginals,
            use_current_marginal_estimate=use_current_marginal_estimate,
            lr=retrain_lr,
            epoch=retrain_epoch,
        )
        result = clf.predict_online_label_shift(shifted_dataloaders)

        # Saving results
        method_name = clf.name
        logger.info(f"Computing {method_name} accuracy")
        accuracies[method_name] = result["accuracy"].item()
        results["detailed_results"][method_name] = result
        results["retrain-lr"] = retrain_lr
        results["retrain-epoch"] = retrain_epoch

        logger.info(f"{method_name} classifier's accuracy: {result['accuracy']}")
        logger.info(f"{method_name} completed in {mscu.tock(start_time):.3g} mins")

    if do_bnadapt and isinstance(base_model, modu.BaseModelWithLinearWeight):
        start_time = mscu.tick()
        clf = modu.BNAdaptClassifier(model=copy.deepcopy(base_model))
        result = clf.predict_online_label_shift(shifted_dataloaders)

        # Saving results
        method_name = clf.name
        logger.info(f"Computing {method_name} accuracy")
        accuracies[method_name] = result["accuracy"].item()
        results["detailed_results"][method_name] = result

        logger.info(f"{method_name} classifier's accuracy: {result['accuracy']}")
        logger.info(f"{method_name} completed in {mscu.tock(start_time):.3g} mins")

    # Collect all marginal estimators for UnsupervisedReweightingClassifiers
    marginal_estimators = []
    if do_rogd or do_all:
        ref_labels = ref_dataset.y_array
        ref_prob = copy.deepcopy(base_model).get_predicted_probabilities(ref_dataloader)
        lrs = [None] if rogd_lrs is None else [None, *[float(lr) for lr in rogd_lrs]]

        for lr in lrs:
            start_time = mscu.tick()
            logger.info("Computing ROGD accuracy")

            # Get ROGD classifier error
            results["rogd-lrs"] = []
            rogd_estimator = modu.RegularOnlineGradientEstimator(
                source_train_marginals=source_train_marginals,
                underlying_estimator=local_shift_estimator,
                ref_labels=ref_labels,
                ref_prob=ref_prob,
                num_labels=num_labels,
                lr=lr,
                marginal_estimator_name="ROGD" if lr is None else f"ROGD_lr={lr}",
                use_smooth_grad=rogd_use_smooth,
            )
            if lr is None:
                if use_default_rogd_lipschitz:
                    rogd_estimator.lipschitz = prju.ModelParameters.rogd_lipschitz[
                        (data_name, base_model_name)
                    ]
                rogd_estimator.estimate_learning_rate(
                    args.total_time,
                    lipschitz_ntries=100,
                    seed=seed,
                )
                results["rogd-lrs"].append(rogd_estimator.lr)
            marginal_estimators.append(rogd_estimator)

    if do_fth or do_all:
        # Get FTH classifier error
        fth_estimator = modu.FollowTheHistoryEstimator(
            source_train_marginals=source_train_marginals,
            underlying_estimator=local_shift_estimator,
        )
        marginal_estimators.append(fth_estimator)

    if do_ftfwh or do_all:
        # Get FTFWH classifier error
        ftfwh_estimator = modu.FollowTheFixedWindowHistoryEstimator(
            source_train_marginals=source_train_marginals,
            underlying_estimator=local_shift_estimator,
            window_size=100,
        )
        marginal_estimators.append(ftfwh_estimator)

    if do_flhftl or do_all:
        flhftl_estimator = modu.FollowLeadingHistoryFollowTheLeaderEstimator(
            source_train_marginals=source_train_marginals,
            underlying_estimator=local_shift_estimator,
            num_labels=num_labels,
        )
        marginal_estimators.append(flhftl_estimator)

    if do_simp_local or do_all:
        delayed_estimator = modu.SimpleLocalShiftEstimator(
            source_train_marginals=source_train_marginals,
            underlying_estimator=local_shift_estimator,
        )
        marginal_estimators.append(delayed_estimator)

    if do_marg_oracle or do_all:
        marg_oracle_estimator = modu.OracleEstimator(
            source_train_marginals=source_train_marginals,
            underlying_estimator=local_shift_estimator,
            num_labels=num_labels,
        )
        marginal_estimators.append(marg_oracle_estimator)

    # Do predictions for all UnsupervisedReweightingClassifiers
    for marginal_estimator in marginal_estimators:
        start_time = mscu.tick()
        clf = modu.UnsupervisedReweightingClassifier(
            model=copy.deepcopy(base_model),
            marginal_estimator=marginal_estimator,
            source_train_marginals=source_train_marginals,
            num_labels=num_labels,
            use_current_marginal_estimate=use_current_marginal_estimate,
        )
        logger.info(f"Computing {clf.name} accuracy")
        result = clf.predict_online_label_shift(shifted_dataloaders)

        # Saving results
        method_name = clf.name
        accuracies[method_name] = result["accuracy"].item()
        results["detailed_results"][method_name] = result

        logger.info(f"{clf.name} classifier's accuracy: {result['accuracy']}")
        logger.info(
            f"{clf.name} classifier's mean marg error: {np.mean(result['l2-sq_marginal_errors'])}"
        )
        logger.info(f"{clf.name} completed in {mscu.tock(start_time):.3g} mins")

    return results


if __name__ == "__main__":
    args = get_parser().parse_args()
    set_marginals(args)

    # Initialize folder to save results to
    if args.save:
        folder = slu.get_result_folder(params=vars(args), root_path=init_path.parent_path)
        file_handler = FileHandler(filename=f"{folder}/output.log")
        logger = mscu.get_logger(args.log_level, file_handler)
    else:
        logger = mscu.get_logger(args.log_level)
    logger.info(f"Experiment parameters: {vars(args)}")

    logger.info(f"Setting seed to be {args.seed}")
    mscu.set_seed(args.seed)

    # Save experiment results & parameters
    results = run_online_label_shift_experiment(
        data_name=args.data_name,
        base_model_name=args.base_model_name,
        shift_type=args.shift_type,
        marg1=args.marg1,
        marg2=args.marg2,
        num_online_samples=args.num_online_samples,
        batch_size=args.batch_size,
        total_time=args.total_time,
        logger=logger,
        seed=args.seed,
        source_test_ratio=args.source_test_ratio,
        use_source_test=args.use_source_test,
        do_ofc=args.do_ofc,
        ofc_ntries=args.ofc_ntries,
        do_rogd=args.do_rogd,
        use_default_rogd_lipschitz=args.use_default_rogd_lipschitz,
        rogd_use_smooth=args.rogd_use_smooth,
        rogd_lrs=args.rogd_lrs.split(" ") if args.rogd_lrs is not None else args.rogd_lrs,
        do_fth=args.do_fth,
        do_ftfwh=args.do_ftfwh,
        do_uogd=args.do_uogd,
        uogd_lrs=args.uogd_lrs.split(" ") if args.uogd_lrs is not None else args.uogd_lrs,
        do_atlas=args.do_atlas,
        do_flhftl=args.do_flhftl,
        do_simp_local=args.do_simp_local,
        do_marg_oracle=args.do_marg_oracle,
        do_retrain=args.do_retrain,
        retrain_lr=args.retrain_lr,
        retrain_epoch=args.retrain_epoch,
        do_bnadapt=args.do_bnadapt,
        do_all=args.do_all,
        calibration_type=args.calibration_type,
        shift_estimator_type=args.shift_estimator_type,
        use_current_marginal_estimate=args.use_current_marginal_estimate,
    )
    results.update(vars(args))

    detailed_results = results.pop("detailed_results")
    logger.info(f"Results: {results}")
    if args.save:
        logger.info(f"Saving results to {folder}")

        # Saving numpy array results separately
        for method in detailed_results:
            save_ols_results(detailed_results[method], folder, method)

        # Saving experiment's summary
        with open(folder / "results.json", mode="w") as f:
            json.dump(mscu.serialize_dict(results), f, indent=2)
