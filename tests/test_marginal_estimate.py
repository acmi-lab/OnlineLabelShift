import argparse

import init_path
import numpy as np
from torch.utils.data import DataLoader

import utils.data_utils as datu
import utils.label_shift_utils as lsu
import utils.model_utils as modu


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
        "-s", "--shift-type", default="monotone", type=str, required=False,
    )
    parser.add_argument("--marg1", type=str, required=False, default=None)
    parser.add_argument("--marg2", type=str, required=False, default=None)
    parser.add_argument(
        "-n", "--num-online-samples", default=10, type=int, required=False
    )
    parser.add_argument("-b", "--batch-size", default=32, type=int, required=False)
    parser.add_argument("-t", "--total-time", default=10, type=int, required=False)
    parser.add_argument("--seed", default=2121, type=int, required=False)
    return parser


def main(
    data_name,
    base_model_name,
    shift_type,
    marg1,
    marg2,
    num_online_samples,
    batch_size,
    total_time,
):
    # Get model and data
    model_cls = modu.get_base_model_class(data_name, base_model_name)
    model = model_cls(pretrained=True, root_path=init_path.parent_path)
    datasets = datu.get_datasets(
        dataname=data_name, source=True, target=True, root_path=init_path.parent_path
    )
    num_labels = datu.DATA_NUM_LABELS[args.data_name]
    shifts = lsu.get_shifts(shift_type, marg1=marg1, marg2=marg2, total_time=total_time)
    shifted_dataloaders = lsu.get_shifted_dataloaders(
        datasets["target"], shifts, num_online_samples, batch_size, num_labels=num_labels
    )

    # Prepare naive bbse components
    src_test_dataloader = DataLoader(
        datasets["source_test"], batch_size=batch_size, shuffle=True
    )
    source_train_marginals = lsu.get_label_marginals(datasets["source_train"].y_array)
    source_train_num_data = datasets["source_train"].y_array.shape[0]
    source_y_pred, source_y_true = model.get_predictions(src_test_dataloader, verbose=True)
    confusion_matrix = lsu.get_confusion_matrix(source_y_true, source_y_pred, num_labels)

    # Get Shift estimators
    bbse = modu.BlackBoxShiftEstimator(
        model=model,
        source_train_marginals=source_train_marginals,
        num_labels=num_labels,
        ref_dataloader=src_test_dataloader,
    )
    rlls = modu.RegularizedShiftEstimator(
        model=model,
        source_train_marginals=source_train_marginals,
        source_train_num_data=source_train_num_data,
        num_labels=num_labels,
        ref_dataloader=src_test_dataloader,
    )

    print("*" * 42)
    for dataloader in shifted_dataloaders:
        y_pred, y_true = model.get_predictions(dataloader)
        y_pred_marginals = lsu.get_label_marginals(y_pred, num_labels)

        # Estimate shift via BBSE
        bbse_marginal_est = bbse._get_marginal_estimate(y_pred_marginals)

        # Estiamte shift via RLLS
        rlls_marginal_est = rlls._get_marginal_estimate(y_pred_marginals)

        print(f"Labels: {y_true}")
        print(f"Real label marginal: {lsu.get_label_marginals(y_true, num_labels)}")
        print(f"Pred label marginal: {y_pred_marginals}")
        print(f"BBSE marginal estimate: {bbse_marginal_est}")
        print(f"BBSE marginal estimate sum: {np.sum(bbse_marginal_est)}")
        print(f"RLLS marginal estimate: {rlls_marginal_est}")
        print(f"RLLS marginal estimate sum: {np.sum(rlls_marginal_est)}")
        print("*" * 42)


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


if __name__ == "__main__":
    args = get_parser().parse_args()
    set_marginals(args)

    # Save experiment results & parameters
    results = main(
        data_name=args.data_name,
        base_model_name=args.base_model_name,
        shift_type=args.shift_type,
        marg1=args.marg1,
        marg2=args.marg2,
        num_online_samples=args.num_online_samples,
        batch_size=args.batch_size,
        total_time=args.total_time,
    )
