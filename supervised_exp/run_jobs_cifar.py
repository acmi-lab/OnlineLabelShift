import json
import os
import sys
import time
from collections import Counter
from datetime import date
from subprocess import Popen

def check_for_done(l):
    for i, p in enumerate(l):
        if p.poll() is not None:
            return True, i
    return False, False

NUM_RUNS=8
GPU_IDS=[6,7]
NUM_GPUS=len(GPU_IDS)
counter=0

DATASETS = [
    'cifar10',
    # 'cifar100',
]
ALGORITHMS=["CT", "RS-CT"]
CALIBRATE=["False"]
# "RS-RT", "CT", "RT"]
LABEL_SHIFT_TYPE=["bernouli", "square", "sinusoidal", "monotone"]


procs = list()
gpu_queue = list()
gpu_use = list()

for i in range(NUM_RUNS):
    gpu_queue.append(GPU_IDS[i % NUM_GPUS])
    
for dataset in DATASETS:
    for algorithm in ALGORITHMS: 
        for label_shift_type in LABEL_SHIFT_TYPE: 
            for calibrate in CALIBRATE: 
                gpu_id = gpu_queue.pop(0)
                gpu_use.append(gpu_id)

                cmd=f"CUDA_VISIBLE_DEVICES={gpu_id} python run_expt.py --dataset {dataset} --root_dir ./data \
                --seed 42 --pretrained False --transform image_none --additional_train_transform randaugment --algorithm  {algorithm}\
                --target_resolution 32 --resize_resolution 32 --default_normalization False --mean 0.5074 0.4867 0.4411 --std 0.2011 0.1987 0.2025 \
                --num_time_steps 1000 --label_shift_type {label_shift_type} --alpha 3.0 --label_shift_kwargs marg1=None marg2=None --calibrate {calibrate}\
                --model resnet18 --lr 0.1 --optimizer SGD --weight_decay 0.0001 --optimizer_kwargs momentum=0.9 --batch_size 200 --progress_bar"
                
                print(cmd)                    
                procs.append(Popen(cmd, shell=True))
                
                time.sleep(3)

                counter += 1

                if len(procs) == NUM_RUNS:
                    wait = True
                    
                    while wait: 
                        done, num = check_for_done(procs)
                    
                        if done: 
                            procs.pop(num)
                            gpu_queue.append(gpu_use.pop(num))
                            wait = False
                        else:
                            time.sleep(3)    
                        
                    print("\n \n \n \n --------------------------- \n \n \n \n")
                    print(f"{date.today()} - {counter} runs completed")
                    sys.stdout.flush()
                    print("\n \n \n \n --------------------------- \n \n \n \n")


for p in procs:
    p.wait()
procs = []