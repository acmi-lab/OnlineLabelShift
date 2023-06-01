import argparse

import init_path
import torch.nn as nn
import torch.optim as optim
from init_path import parent_path
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils.data_utils as datu
import utils.misc_utils as mscu
import utils.model_utils as modu
from utils.proj_utils import Parameters

CRITERIONS = {"CrossEntropy": nn.CrossEntropyLoss()}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-name", type=str, default="synthetic")
    parser.add_argument("-m", "--model-name", type=str, default="logreg")
    parser.add_argument("-e", "--num-epoch", type=int, default=50)
    parser.add_argument("-b", "--batch-size", type=int, default=200)
    parser.add_argument("-l", "--lr", type=float, default=0.1)
    parser.add_argument("--log-level", default="INFO", required=False)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--random-seed", type=int, default=4242)
    parser.add_argument("--display-iterations", type=int, default=150)
    parser.add_argument(
        "--criterion-name", type=str, default="CrossEntropy", choices=CRITERIONS.keys()
    )
    parser.add_argument("--scheduler-gamma", type=float, default=0.1)
    parser.add_argument("--scheduler-milestones", type=str, default="25 40")
    parser.add_argument("--weight-lipschitz-tries", type=int, default=1_000)
    parser.add_argument("--step-limit", type=int, default=None)
    return parser


def training_loop(
    model,
    optimizer,
    scheduler,
    train_dataloader,
    test_dataloader,
    display_iterations,
    num_epoch,
    criterion,
    step_limit,
):
    model.train()

    step_count = 0
    metrics = {
        "train_loss": [],
        "test_loss": [],
        "train_accuracy": [],
        "test_accuracy": [],
    }
    for epoch in range(num_epoch):
        for batch_idx, (X, y) in tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Epoch {epoch}",
        ):
            X, y = X.to(Parameters.device), y.to(Parameters.device)

            # GD
            optimizer.zero_grad()
            logits_pred = model(X)
            loss = criterion(logits_pred, y)
            loss.backward()
            optimizer.step()

            # Evaluate metrics
            if step_count % display_iterations == display_iterations - 1:
                train_loss, train_acc = model.get_loss_and_accuracy(
                    train_dataloader, criterion
                )
                test_loss, test_acc = model.get_loss_and_accuracy(
                    test_dataloader, criterion
                )

                print(
                    f"Epoch {epoch}"
                    f" Step {step_count}"
                    f" Train Loss: {train_loss:.7f}"
                    f" Test Loss: {test_loss:.7f}"
                    f" Train Acc: {train_acc:03.3f}"
                    f" Test Acc: {test_acc:03.3f}"
                )

                metrics["train_loss"].append(train_loss)
                metrics["train_accuracy"].append(train_acc)
                metrics["test_loss"].append(test_loss)
                metrics["test_accuracy"].append(test_acc)
            step_count += 1
            if step_limit and step_limit <= step_count:
                return metrics
        scheduler.step()
    return metrics


if __name__ == "__main__":
    print(f"Training base classifier with {Parameters.device}")
    args = get_parser().parse_args()
    print(f"Training parameters: {vars(args)}")

    logger = mscu.get_logger(args.log_level)
    mscu.set_seed(args.random_seed)

    # Get model and data
    model_cls = modu.get_base_model_class(args.data_name, args.model_name)
    datasets = datu.get_datasets(
        dataname=args.data_name, source=True, target=True, root_path=parent_path
    )
    train_dataloader = DataLoader(
        datasets["source_train"], batch_size=args.batch_size, shuffle=True,
    )
    test_dataloader = DataLoader(
        datasets["target"], batch_size=args.batch_size, shuffle=True,
    )

    if issubclass(model_cls, modu.BaseModelNN):
        model = model_cls(pretrained=False, root_path=parent_path)

        # Setup optimizer
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            gamma=args.scheduler_gamma,
            milestones=args.scheduler_milestones.split(" "),
        )

        criterion = CRITERIONS[args.criterion_name]
        metrics = training_loop(
            model,
            optimizer,
            scheduler,
            train_dataloader,
            test_dataloader,
            display_iterations=args.display_iterations,
            num_epoch=args.num_epoch,
            criterion=criterion,
            step_limit=args.step_limit,
        )

        # Save model
        model.save()

        # Compute lipschitz estimate required by UOGD
        model.sample_linear_weight_lipschitz(criterion, num_tries=args.weight_lipschitz_tries)

        details = {}
        details.update(metrics)
        details.update(vars(args))
        model.save_details(details)
    elif issubclass(model_cls, modu.BaseModelRandomForest):
        rf_params = {
            "n_estimators": 100,
            "max_depth": 2,
        }
        # Training random forest
        source_train_X, source_train_y = mscu.pytorch2sklearn(train_dataloader)
        model = model_cls(pretrained=False, root_path=parent_path, **rf_params)
        model.rf_clf.fit(source_train_X, source_train_y)
        model.save()


