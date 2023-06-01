# Online Label Shift


`OnlineLabelShift` is the official implementation of the accompanying paper [Online Label Shift: Optimal Dynamic Regret meets Practical Algorithms](https://arxiv.org/pdf/2305.19570.pdf). 
For more details, please refer to the paper. 

<p align="center">
  <img src="https://github.com/acmi-lab/OnlineLabelShift/assets/43626310/3be87ecf-641e-49d7-92a8-993825fabbeb" />
</p>

## Setup Environment

```
conda env update --file environment.yml
```

## Run Online Label Shift Experiment

The following command runs the online label shift experiment.
It expects the base model has been trained and saved under `/model`

```
python scripts/run_ols.py -d synthetic -m logreg --do-all 1 -t 1000 --save 1
```

To see all the options

```
python scripts/run_ols.py -h
```

## Train model

This script supports model training for synthetic, cifar10, and mnist datasets.

```
python scripts/train_model.py -d <data> -m <model> -e <epoch>
```

The corresponding models are:

| Data      | Model    |
| --------- | -------- |
| synthetic | logreg   |
| mnist     | fcn      |
| cifar10   | resnet18 |

## Generate Synthetic Data

To run experiments on synthetic data, one should first generate the data with the following command:

```
python scripts/gen_synth_data.py
```
