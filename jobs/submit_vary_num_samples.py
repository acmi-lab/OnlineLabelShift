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
    parser.add_argument(
        "-n",
        "--nums_samples",
        default="1 100",
        type=str,
        required=False,
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
    parser.add_argument("--num-runs", default=12, type=int, required=False)
    return parser

def submit_vary_num_samples(
    data_name,
    model_name,
    nums_samples,
    cuda_device_id,
    seeds,
    shift,
):
    cuda_prefix = f"CUDA_VISIBLE_DEVICES={cuda_device_id}"
    prefix = f"{cuda_prefix} nohup"
    postfix = "&"

    for num_samples in nums_samples:
        for seed in seeds:
            command = (
                f"python run_ols.py -d {data_name} -m {model_name}"
                f" -t 1000 --seed {seed} -s {shift} --num-online-samples {num_samples}"
                " --do-all 1 --save 1"
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

    submit_vary_num_samples(
        data_name=args.data_name,
        model_name=args.base_model_name,
        nums_samples=args.nums_samples.split(" "),
        cuda_device_id=args.cuda_device_id,
        seeds=args.seeds.split(" "),
        shift=args.shift,
    )
