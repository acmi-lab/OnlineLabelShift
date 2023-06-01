import argparse
from subprocess import Popen
import time
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
        "--shift",
        default="monotone",
        type=str,
        required=False,
    )
    parser.add_argument("--seeds", default="8610 1308 2011", type=str, required=False)
    parser.add_argument("--uogd-lrs", default="0.01 0.05 0.1 0.2 0.4 0.8 1.6", type=str, required=False)
    return parser

def submit_vary_uogd_lr(
    data_name,
    model_name,
    uogd_lrs,
    cuda_device_id,
    seeds,
    shift,
):
    cuda_prefix = f"CUDA_VISIBLE_DEVICES={cuda_device_id}"
    prefix = f"{cuda_prefix} nohup"
    postfix = "&"

    for seed in seeds:
        command = (
            f"python run_ols.py -d {data_name} -m {model_name}"
            f" -t 1000 --seed {seed} -s {shift}"
            f" --uogd-lrs '{uogd_lrs}' --do-uogd 1 --save 1"
        )
        print(f"Sumitting command: {command}")
        cur_proc = Popen(
            f"{prefix} {command} {postfix}",
            cwd=init_path.parent_path + "/scripts",
            shell=True,
        )
        time.sleep(10)

if __name__ == "__main__":
    args = get_parser().parse_args()

    submit_vary_uogd_lr(
        data_name=args.data_name,
        model_name=args.base_model_name,
        uogd_lrs=args.uogd_lrs,
        cuda_device_id=args.cuda_device_id,
        seeds=args.seeds.split(" "),
        shift=args.shift,
    )
