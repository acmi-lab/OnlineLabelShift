"""This script tests the implementation of sampling"""
import argparse

import init_path
import numpy as np

import utils.label_shift_utils as lsu
import utils.misc_utils as mscu


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--num-labels", default=3, type=int, required=False)
    parser.add_argument("-n", "--num-samples", default=100, type=int, required=False)
    parser.add_argument("-i", "--num-iterations", default=10, type=int, required=False)
    parser.add_argument("-d", "--num-data", default=10000, type=int, required=False)
    parser.add_argument("-s", "--seed", default=42, type=int, required=False)
    return parser


def main(num_labels, num_samples, num_iterations, num_data, seed):
    # Create random labels
    mscu.set_seed(seed)
    labels = np.random.randint(low=0, high=num_labels, size=num_data)

    base_marginals = lsu.get_label_marginals(labels, num_labels=num_labels)
    print(f"Overall marginal: {base_marginals}")
    print("*" * 84)
    for _ in range(num_iterations):
        target_marginals = lsu.get_random_probability(num_labels)
        target_idc = lsu.sample_idc_wo_replacement(
            labels, num_samples, target_marginals, num_labels=num_labels
        )
        num_unique_idc = len(set(target_idc))

        # Compute current marginals
        cur_labels = [labels[idx] for idx in target_idc]
        cur_marginals = lsu.get_label_marginals(cur_labels, num_labels=num_labels)
        print(f"Target marginals: {target_marginals}")
        print(f"Num samples: {num_samples}. Num unique indicies: {num_unique_idc}")
        print(f"Current marginals: {cur_marginals}")
        print("*" * 84)


if __name__ == "__main__":
    args = get_parser().parse_args()
    print(f"Experiment parameters: {vars(args)}")

    # Save experiment results & parameters
    results = main(
        num_labels=args.num_labels,
        num_samples=args.num_samples,
        num_iterations=args.num_iterations,
        num_data=args.num_data,
        seed=args.seed,
    )
