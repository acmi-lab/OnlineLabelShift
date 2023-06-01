import argparse
import sys
import time
from datetime import date
from subprocess import Popen

import init_path


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data-name", default="synthetic", type=str, required=False,
    )
    parser.add_argument(
        "-m", "--base-model-name", default="logreg", type=str, required=False
    )
    parser.add_argument("-c", "--cuda-device-id", default=0, type=int, required=False)
    parser.add_argument(
        "-s",
        "--shifts",
        default="monotone square sinusoidal bernouli",
        type=str,
        required=False,
    )
    parser.add_argument("--seeds", default="8610 1308 2011", type=str, required=False)
    parser.add_argument("--num-runs", default=12, type=int, required=False)
    return parser


def submit_ols(data_name, model_name, cuda_device_id, seeds, shifts, num_runs):
    cuda_prefix = f"CUDA_VISIBLE_DEVICES={cuda_device_id}"
    prefix = f"{cuda_prefix} nohup"
    postfix = "&"

    num_jobs_ran = 0
    procs = []
    for shift in shifts:
        for seed in seeds:
            command = (
                f"python run_ols.py -d {data_name} -m {model_name} --do-calibration 1"
                f" -t 1000 --seed {seed} -s {shift} --do-all 1 --save 1"
            )
            print(f"Sumitting command: {command}")
            cur_proc = Popen(
                f"{prefix} {command} {postfix}",
                cwd=init_path.parent_path + "/scripts",
                shell=True,
            )
            procs.append(cur_proc)
            time.sleep(10)

            num_jobs_ran += 1

            if num_jobs_ran % num_runs == 0:
                for p in procs:
                    p.wait()
                procs = []
                time.sleep(10)

                print("\n \n \n \n --------------------------- \n \n \n \n")
                print(f"{date.today()} - {num_jobs_ran} runs submitted")
                sys.stdout.flush()
                print("\n \n \n \n --------------------------- \n \n \n \n")


if __name__ == "__main__":
    args = get_parser().parse_args()

    # Parameters
    seeds = [8610, 1308, 2011]
    shifts = [
        "monotone",
        "square",
        "sinusoidal",
        "bernouli",
    ]

    submit_ols(
        data_name=args.data_name,
        model_name=args.base_model_name,
        cuda_device_id=args.cuda_device_id,
        seeds=args.seeds.split(" "),
        shifts=args.shifts.split(" "),
        num_runs=args.num_runs,
    )
