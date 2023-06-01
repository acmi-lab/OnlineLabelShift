import copy
import json
import logging
import pickle

import calibration as cal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from transformers import DistilBertForSequenceClassification, DistilBertModel

import utils.data_utils as datu
import utils.label_shift_utils as lsu
import utils.misc_utils as mscu
import utils.proj_utils as prju

logger = logging.getLogger("online_label_shift")


### Base models
class BaseModel:
    def __init__(self, root_path, model_name) -> None:
        super().__init__()
        self.model_folder = prju.Parameters.get_model_path(root_path=root_path)
        self.state_dict_path = self.model_folder + f"/{model_name}.pth"
        self.details_path = self.model_folder + f"/{model_name}.json"

    def save_details(self, training_details):
        with open(self.details_path, "w") as f:
            json.dump(training_details, f, indent=2)

    def get_confusion_matrix(self, dataloader, num_labels):
        y_pred, y_true = self.get_predictions(dataloader)
        confusion_matrix = lsu.get_confusion_matrix(y_true, y_pred, num_labels)
        return confusion_matrix

    def get_soft_confusion_matrix(self, dataloader, num_labels):
        dataloader.shuffle = False

        prob, y_true = self.get_predicted_probabilities(dataloader, return_labels=True)
        soft_confusion_matrix = lsu.get_soft_confusion_matrix(y_true, prob, num_labels)
        dataloader.shuffle = True
        return soft_confusion_matrix

    def save(self, training_details=None):
        pass

    def get_predictions(self, dataloader, verbose=False):
        pass

    def get_predicted_probabilities(self, dataloader, verbose=False, return_labels=False):
        pass


class BaseModelNN(BaseModel, nn.Module):
    def __init__(self, root_path, model_name) -> None:
        super().__init__(root_path, model_name)

    def save(self, training_details=None):
        torch.save(self.state_dict(), self.state_dict_path)
        if training_details is not None:
            self.save_details(training_details)

    def get_loss_and_accuracy(self, dataloader, criterion):
        self.eval()
        with torch.no_grad():
            total_loss = 0
            total_correct = 0
            for _, (X, y) in enumerate(dataloader):
                X, y = X.to(prju.Parameters.device), y.to(prju.Parameters.device)

                # Get loss
                logits_pred = self.forward(X)
                batch_loss = criterion(logits_pred, y)
                total_loss += torch.sum(batch_loss).data.cpu().numpy()

                # Get # of correct predictions
                y_hat = torch.argmax(logits_pred, dim=1)
                n_correct = torch.sum(y_hat == y).data.cpu().numpy()
                total_correct += n_correct
        self.train()

        n_samples = len(dataloader.dataset)
        avg_loss = total_loss / n_samples
        accuracy = total_correct / n_samples
        return avg_loss, accuracy

    def get_predictions(self, dataloader, verbose=False):
        self.eval()
        with torch.no_grad():
            y_true = torch.zeros((0))
            y_pred = torch.zeros((0))

            iterator = tqdm(dataloader, total=len(dataloader)) if verbose else dataloader
            for X, y in iterator:
                X, y = X.to(prju.Parameters.device), y.to(prju.Parameters.device)

                cur_logits_pred = self.forward(X)
                cur_y_pred = torch.argmax(cur_logits_pred, dim=1)
                y_true = torch.cat((y_true, y.flatten().cpu()))
                y_pred = torch.cat((y_pred, cur_y_pred.cpu()))
        self.train()

        return y_pred, y_true

    def get_predicted_probabilities(self, dataloader, verbose=False, return_labels=False):
        self.eval()
        with torch.no_grad():
            probabilities = torch.zeros((0))
            if return_labels:
                y_true = torch.zeros((0))

            iterator = tqdm(dataloader, total=len(dataloader)) if verbose else dataloader
            for X, y in iterator:
                X, y = X.to(prju.Parameters.device), y.to(prju.Parameters.device)

                cur_logits_pred = self.forward(X)
                cur_probabilities = nn.Softmax(dim=1)(cur_logits_pred)
                probabilities = torch.cat((probabilities, cur_probabilities.cpu()))

                if return_labels:
                    y_true = torch.cat((y_true, y.flatten().cpu()))
        self.train()

        if not return_labels:
            return probabilities
        else:
            return probabilities, y_true


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.expansion = 1
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, features=False):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.features = features
        if not self.features:
            self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if self.features:
            return out
        else:
            final_out = self.linear(out)
            return final_out


def ResNet18(num_classes=10, features=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, features=features)


class BaseModelWithLinearWeight(BaseModelNN):
    def __init__(
        self,
        pretrained,
        root_path,
        model_name,
        input_size,
        num_labels,
        feature_extractor: nn.Module,
        feature_dim,
        calibrator=None,
    ) -> None:
        super().__init__(root_path, model_name)

        self.input_size = input_size
        self.num_labels = num_labels
        self.feature_extractor = feature_extractor
        self.feature_dim = feature_dim
        self.linear = nn.Linear(self.feature_dim, self.num_labels, bias=True)

        self.linear.bias = nn.Parameter(torch.zeros(self.num_labels), requires_grad=False)

        self.device = prju.Parameters.device
        self.linear.to(self.device)
        self.feature_extractor.to(self.device)

        self.weight_lipschitz_estimate = -1e5 * torch.ones(1, device=self.device)
        if pretrained:
            ckpt = torch.load(self.state_dict_path, map_location=self.device)

            if "algorithm" in ckpt:
                ckpt = ckpt["algorithm"]

            state_dict = {}
            for k, v in ckpt.items():
                if k.startswith("model.1"):
                    k = f"linear{k[7:]}"
                elif k.startswith("model.0"):
                    k = f"feature_extractor{k[7:]}"

                state_dict[k] = v

            self.load_state_dict(state_dict, strict=False)

            try:
                with open(self.details_path, "r") as f:
                    details = json.load(f)
                self.weight_lipschitz_estimate = details["weight_lipschitz_estimate"]
            except:
                pass
                self.weight_lipschitz_estimate = 1.0

        self.calibrator = calibrator

    def sample_linear_weight_lipschitz(
        self, criterion, num_tries=500, weight_name="linear.weight"
    ):
        self.train()

        # Store model weights
        param_norms = {}
        param_temps = {}
        for name, param in self.named_parameters():
            if name != weight_name:
                continue
            param_norms[name] = torch.linalg.norm(param.data)
            param_temps[name] = param.data.clone()

        # Try random W, X, and y for lipschitz constant for model weights
        for _ in tqdm(range(num_tries), desc="Finding linear layer Lipschitz"):
            for name, param in self.named_parameters():
                if name != weight_name:
                    continue
                if param.grad is not None:
                    param.grad.zero_()
                param.requires_grad_(False)
                rand_weight = torch.randn(size=param.data.shape, device=self.device)
                rand_norm = np.random.random() * param_norms[name]
                rand_weight = rand_weight / torch.linalg.norm(rand_weight) * rand_norm
                param.copy_(rand_weight)
                param.requires_grad_(True)
            rand_X = torch.randn(
                size=(
                    1,
                    *self.input_size,
                ),
                device=self.device,
            )
            rand_y = torch.randint(
                low=0, high=self.num_labels, size=(1,), device=self.device
            )
            rand_logits_pred = self.forward(rand_X)
            loss = criterion(rand_logits_pred, rand_y)
            loss.backward()
            weight_lipschitz = torch.linalg.norm(self.linear.weight.grad)
            self.update_weight_lipschitz_estimate(weight_lipschitz)

        # Restore model weights
        for name, param in self.named_parameters():
            if name != weight_name:
                continue
            param.requires_grad_(False)
            param.copy_(param_temps[name])
            param.requires_grad_(True)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.linear(x)
        if self.calibrator is not None:
            x = self.calibrator.calibrate(x)
        return x

    def update_weight_lipschitz_estimate(self, new_estimate):
        # logger.debug(f"New weight lipschitz: {new_estimate}")
        if new_estimate > self.weight_lipschitz_estimate:
            logger.info(f"Updating with: {new_estimate}")
            self.weight_lipschitz_estimate = new_estimate

    def save_details(self, details):
        details["weight_lipschitz_estimate"] = self.weight_lipschitz_estimate.item()
        return super().save_details(details)


class Resnet18_CIFAR10(BaseModelWithLinearWeight):
    def __init__(self, pretrained=False, root_path="../", model_name="resnet18_cifar"):
        feature_extractor = ResNet18(features=True)
        super().__init__(
            pretrained,
            root_path,
            model_name,
            input_size=(3, 32, 32),
            num_labels=10,
            feature_extractor=feature_extractor,
            feature_dim=512,
        )


class Resnet18_CINIC(BaseModelWithLinearWeight):
    def __init__(self, pretrained=False, root_path="../", model_name="resnet18_cinic"):
        feature_extractor = ResNet18(features=True)
        super().__init__(
            pretrained,
            root_path,
            model_name,
            input_size=(3, 32, 32),
            num_labels=10,
            feature_extractor=feature_extractor,
            feature_dim=512,
        )


class MLP_Fashion(BaseModelWithLinearWeight):
    def __init__(self, pretrained=False, root_path="../", model_name="MLP_fashion"):
        feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 100, bias=True),
            nn.ReLU(),
            nn.Linear(100, 100, bias=True),
            nn.ReLU(),
            nn.Linear(100, 100, bias=True),
            nn.ReLU(),
        )

        super().__init__(
            pretrained,
            root_path,
            model_name,
            input_size=(784,),
            num_labels=10,
            feature_extractor=feature_extractor,
            feature_dim=100,
        )


def initialize_torchvision_model(name, d_out, pretrained=True):
    import antialiased_cnns

    # get constructor and last layer names
    if name == "wideresnet50":
        constructor_name = "wide_resnet50_2"
        last_layer_name = "fc"
    elif name == "densenet121":
        constructor_name = name
        last_layer_name = "classifier"
    elif name in ("resnet18", "resnet34", "resnet50", "resnet101"):
        constructor_name = name
        last_layer_name = "fc"
    elif name in ("efficientnet_b0"):
        constructor_name = name
        last_layer_name = "classifier"
    else:
        raise ValueError(f"Torchvision model {name} not recognized")
    # construct the default model, which has the default last layer
    constructor = getattr(antialiased_cnns, constructor_name)
    model = constructor(pretrained=pretrained)
    # adjust the last layer
    d_features = getattr(model, last_layer_name).in_features
    if d_out is None:  # want to initialize a featurizer model
        last_layer = nn.Identity(d_features)
        model.d_out = d_features
    else:  # want to initialize a classifier for a particular num_classes
        last_layer = nn.Linear(d_features, d_out)
        model.d_out = d_out
    setattr(model, last_layer_name, last_layer)

    return model


class Resnet18_EuroSAT(BaseModelWithLinearWeight):
    def __init__(self, pretrained=False, root_path="../", model_name="resnet18_eurosat"):
        feature_extractor = initialize_torchvision_model(
            name="resnet18", d_out=None, pretrained=False
        )

        super().__init__(
            pretrained,
            root_path,
            model_name,
            input_size=(3, 64, 64),
            num_labels=10,
            feature_extractor=feature_extractor,
            feature_dim=feature_extractor.d_out,
        )


class MLP_SHL(BaseModelWithLinearWeight):
    def __init__(self, pretrained=False, root_path="../", model_name="MLP_shl"):
        feature_extractor = nn.Identity()

        super().__init__(
            pretrained,
            root_path,
            model_name,
            input_size=(22,),
            num_labels=6,
            feature_extractor=feature_extractor,
            feature_dim=22,
        )


class FCN_CIFAR10(BaseModelWithLinearWeight):
    def __init__(self, pretrained=False, root_path="../", model_name="fcn_cifar"):
        feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 20),
            nn.ReLU(),
        )
        super().__init__(
            pretrained,
            root_path,
            model_name,
            input_size=(3, 32, 32),
            num_labels=10,
            feature_extractor=feature_extractor,
            feature_dim=20,
        )


class BERT_ARXIV(BaseModelWithLinearWeight):
    def __init__(self, pretrained=False, root_path="../", model_name="bert_arxiv"):
        feature_extractor = initialize_bert_based_model("distilbert-base-uncased", 23)

        super().__init__(
            pretrained,
            root_path,
            model_name,
            input_size=(512, 2),
            num_labels=23,
            feature_extractor=feature_extractor,
            feature_dim=feature_extractor.d_out,
        )


class DistilBertClassifier(DistilBertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        return outputs


class Identity(nn.Module):
    """An identity layer"""

    def __init__(self, d):
        super().__init__()
        self.in_features = d
        self.out_features = d

    def forward(self, x):
        return x


class DistilBertFeaturizer(DistilBertModel):
    def __init__(self, config):
        super().__init__(config)
        self.d_out = config.hidden_size

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        hidden_state = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        pooled_output = hidden_state[:, 0]
        return pooled_output


def initialize_bert_based_model(net, num_classes):
    if net == "distilbert-base-uncased":
        model = DistilBertClassifier.from_pretrained(net, num_labels=num_classes)
        d_features = getattr(model, "classifier").in_features

        model.classifier = Identity(d_features)
        model.d_out = d_features
    else:
        raise ValueError(f"Model: {net} not recognized.")
    return model


class LogisticRegression_Synthetic(BaseModelWithLinearWeight):
    def __init__(self, pretrained=False, root_path="../", model_name="logreg_synthetic"):
        feature_extractor = nn.Identity()
        synthetic_name = prju.DatasetParameters.SYNDATA_NAME
        synthetic_defaults = prju.DatasetParameters.dataset_defaults[synthetic_name]

        super().__init__(
            pretrained,
            root_path,
            model_name,
            input_size=(synthetic_defaults["num_dimensions"],),
            num_labels=synthetic_defaults["num_classes"],
            feature_extractor=feature_extractor,
            feature_dim=synthetic_defaults["num_dimensions"],
        )


class FCN_MNIST(BaseModelWithLinearWeight):
    def __init__(
        self,
        pretrained=False,
        root_path="../",
        model_name="fcn_mnist",
    ) -> None:
        feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 20),
            nn.ReLU(),
        )
        super().__init__(
            pretrained,
            root_path,
            model_name,
            input_size=(784,),
            num_labels=10,
            feature_extractor=feature_extractor,
            feature_dim=20,
        )


class FCN_MNIST_EARLY(BaseModelWithLinearWeight):
    def __init__(
        self,
        pretrained=False,
        root_path="../",
        model_name="fcn_mnist_early",
    ) -> None:
        feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 20),
            nn.ReLU(),
        )
        super().__init__(
            pretrained,
            root_path,
            model_name,
            input_size=(784,),
            num_labels=10,
            feature_extractor=feature_extractor,
            feature_dim=20,
        )


class BaseModelRandomForest(BaseModel):
    def __init__(
        self, n_estimators, max_depth, data_name, pretrained, root_path, calibrator=None
    ):
        super().__init__(root_path, model_name=f"randforest_{data_name}")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.rf_clf = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=0
        )
        self.calibrator = calibrator
        self.num_labels = datu.DATA_NUM_LABELS[data_name]

        if pretrained:
            self.load()

    def load(self):
        logger.info(f"Loading Random Forest Classifier from {self.state_dict_path}")
        with open(self.state_dict_path, "rb") as f:
            self.rf_clf = pickle.load(file=f)

    def save(self, training_details=None):
        with open(self.state_dict_path, "wb") as f:
            pickle.dump(self.rf_clf, file=f)

        if training_details is not None:
            self.save_details(training_details)

    def forward(self, X):
        X = torch.flatten(X, start_dim=1)
        X = X.detach().cpu().numpy()
        logits = mscu.prob_to_logit(self.rf_clf.predict_proba(X))
        if self.calibrator is not None:
            logits = self.calibrator.calibrate(logits)
        return logits.to(prju.Parameters.device)

    def get_predictions(self, dataloader, verbose=False):
        X, y = mscu.pytorch2sklearn(dataloader)
        logits = mscu.prob_to_logit(self.rf_clf.predict_proba(X))
        if self.calibrator is not None:
            logits = self.calibrator.calibrate(logits)

        y_pred = np.argmax(logits, axis=1)
        return y_pred, y

    def get_predicted_probabilities(self, dataloader, verbose=False, return_labels=False):
        X, y = mscu.pytorch2sklearn(dataloader)
        logits = mscu.prob_to_logit(self.rf_clf.predict_proba(X))
        if self.calibrator is not None:
            logits = self.calibrator.calibrate(logits)
        logits = mscu.torch2np(logits)
        y_prob = mscu.softmax(logits, axis=1)
        y_prob = mscu.np2torch(y_prob)

        if not return_labels:
            return y_prob
        else:
            return y_prob, y


class MNIST_RandomForest(BaseModelRandomForest):
    def __init__(
        self, pretrained, root_path, n_estimators=100, max_depth=4, calibrator=None
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            data_name="mnist",
            pretrained=pretrained,
            root_path=root_path,
            calibrator=calibrator,
        )


class CIFAR10_RandomForest(BaseModelRandomForest):
    def __init__(
        self, pretrained, root_path, n_estimators=100, max_depth=4, calibrator=None
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            data_name="cifar10",
            pretrained=pretrained,
            root_path=root_path,
            calibrator=calibrator,
        )


def get_base_model_class(data_name, model_name):
    model_constructors = {
        ("mnist", "randforest"): MNIST_RandomForest,
        ("cifar10", "randforest"): CIFAR10_RandomForest,
        ("cifar10", "resnet18"): Resnet18_CIFAR10,
        ("cinic", "resnet18"): Resnet18_CINIC,
        ("eurosat", "resnet18"): Resnet18_EuroSAT,
        ("synthetic", "logreg"): LogisticRegression_Synthetic,
        ("mnist", "fcn"): FCN_MNIST,
        ("mnist", "fcn_early"): FCN_MNIST_EARLY,
        ("cifar10", "fcn"): FCN_CIFAR10,
        ("fashion", "mlp"): MLP_Fashion,
        ("shl", "mlp"): MLP_SHL,
        ("arxiv", "bert"): BERT_ARXIV,
    }
    if (data_name, model_name) in model_constructors:
        return model_constructors[(data_name, model_name)]
    raise ValueError(f"Base model not found for data {data_name} model {model_name}")


### Calibration method
def calibrate_on_torch(self, logits):
    logits = mscu.np2torch(logits)
    logits = self.temperature_scale(logits).detach()
    return logits


def get_calibrator(
    model: BaseModel,
    ref_dataloader,
    device,
    calibration_type,
    print_verbose=False,
):
    def _get_calibrator(device):
        if calibration_type == "ts":
            print(f"Getting TS")
            calibrator = cal.TempScaling(
                bias=False, device=device, print_verbose=print_verbose
            )
        else:
            print(f"Getting VS")
            calibrator = cal.VectorScaling(
                bias=True,
                device=device,
                num_label=model.num_labels,
                print_verbose=print_verbose,
            )
        return calibrator

    if isinstance(model, BaseModelNN):
        calibrator = _get_calibrator(device)
        ref_logits = torch.zeros((0, model.num_labels), device=model.device)
        ref_labels = torch.zeros((0), device=model.device, dtype=torch.int64)

        model.eval()
        with torch.no_grad():
            for X, y in ref_dataloader:
                X, y = X.to(model.device), y.to(model.device)
                cur_logits = model.forward(X)
                ref_logits = torch.cat((ref_logits, cur_logits), dim=0)
                ref_labels = torch.cat((ref_labels, y))
        calibrator.fit(ref_logits.cpu().numpy(), ref_labels.cpu().numpy())
    elif isinstance(model, BaseModelRandomForest):
        calibrator = _get_calibrator("cpu")
        source_test_X, source_test_y = mscu.pytorch2sklearn(ref_dataloader)
        ref_logits = mscu.prob_to_logit(model.rf_clf.predict_proba(source_test_X))
        calibrator.fit(ref_logits, source_test_y.astype(np.long))
    else:
        raise ValueError(f"Calibration not implemented for model of type {type(model)}")

    # Monkey patch calibrator's function to work entirely on torch
    calibrator.calibrate = lambda logits: calibrate_on_torch(calibrator, logits)
    return calibrator


### Classifier
class OptimalFixedReweightClassifier:
    def __init__(self, model: BaseModel, source_train_marginals) -> None:
        self.model = model
        self.source_train_marginals = source_train_marginals
        self.n_labels = source_train_marginals.shape[0]

    def get_accuracy_with_weight(self, logits, y_true, probabilities):
        """Get the accuracy with reweigted predictions
        Args:
            logits (np.array[shape=n_labels]): the logits for target marginals
            y_true (np.array[shape=n_data]): the labels
            probabilities (np.array[shape=(n_data, n_labels)]):
                the probabilities predicted by the model
        """
        target_marginals = mscu.softmax(logits)
        reweighted_pred = lsu.get_reweighted_predictions(
            probabilities, self.source_train_marginals, target_marginals
        )
        return lsu.get_accuracy(y_true, reweighted_pred)

    def get_accuracy(self, dataloader, ntries=30, verbose=False):
        if isinstance(self.model, BaseModelNN):
            self.model.eval()

        # Get predicted probabilities
        prob = torch.zeros((0))
        y_true = torch.zeros((0))
        with torch.no_grad():
            iterator = (
                tqdm(dataloader, desc="Making OFC predictions") if verbose else dataloader
            )
            for X, y in iterator:
                y_true = torch.cat((y_true, y.flatten()))
                X, y = X.to(prju.Parameters.device), y.to(prju.Parameters.device)

                logits = self.model.forward(X)
                cur_prob = nn.Softmax(dim=1)(logits)
                prob = torch.cat((prob, cur_prob.cpu()))
        y_true = y_true.numpy()
        prob = prob.numpy()

        # Optimize target logits
        max_acc = -1
        max_res = None
        iterator = (
            tqdm(range(ntries), desc="Optimizing OFC weight with multiple tries")
            if verbose
            else range(ntries)
        )
        for i in iterator:
            if i == 0:
                logits = np.ones(self.n_labels)
            else:
                logits = np.random.randn(self.n_labels)
            cost_function = lambda x: -self.get_accuracy_with_weight(x, y_true, prob)
            grad_function = lambda x: mscu.finite_gradient(
                cost_function, mscu.softmax(x), delta=5e-3, k=10
            )
            res = minimize(
                cost_function, x0=logits, jac=grad_function, method="L-BFGS-B", tol=1e-5
            )
            if -res.fun > max_acc:
                max_acc = -res.fun
                max_res = res

            print(f"OFC optimiation results:\n{res}")
        if verbose:
            logger.debug(f"OFC optimiation results:\n{max_res}")
        return max_acc


### Marginal Estimators
class MarginalEstimator:
    def __init__(self, source_train_marginals, marginal_estimator_name=None) -> None:
        self.source_train_marginals = source_train_marginals
        self.marginal_estimator_name = marginal_estimator_name

    def get_marginal_estimate(self, dataloader):
        raise NotImplementedError


class LocalShiftEstimator(MarginalEstimator):
    def __init__(
        self,
        model,
        source_train_marginals,
        num_labels,
        ref_dataloader,
        marginal_estimator_name=None,
        confusion_matrix=None,
    ) -> None:
        super().__init__(
            source_train_marginals=source_train_marginals,
            marginal_estimator_name=marginal_estimator_name,
        )
        self.model: BaseModel = model
        if isinstance(self.model, BaseModelNN):
            self.model.eval()
        self.num_labels = num_labels
        self.ref_dataloader = ref_dataloader
        self.confusion_matrix = confusion_matrix
        self.compute_confusion_matrix()

    def compute_confusion_matrix(self):
        if self.confusion_matrix is None:
            self.confusion_matrix = self.model.get_soft_confusion_matrix(
                self.ref_dataloader, self.num_labels
            )

    def get_prediction_marginals(self, dataloader):
        y_pred, _ = self.model.get_predictions(dataloader=dataloader)
        pred_marginals = lsu.get_label_marginals(y_pred, num_labels=self.num_labels)
        return pred_marginals

    def get_soft_prediction_marginals(self, dataloader):
        y_prob = self.model.get_predicted_probabilities(dataloader=dataloader)
        y_prob = mscu.torch2np(y_prob)
        pred_marginals = np.mean(y_prob, axis=0)
        return pred_marginals

    def get_marginal_estimate(self, dataloader):
        pred_marginals = self.get_soft_prediction_marginals(dataloader)
        return self._get_marginal_estimate(pred_marginals=pred_marginals)

    def _get_marginal_estimate(self, pred_marginals):
        raise NotImplementedError


class ShiftEstimatorWithMemory(MarginalEstimator):
    def __init__(
        self,
        source_train_marginals,
        underlying_estimator: LocalShiftEstimator,
        marginal_estimator_name,
    ) -> None:
        super().__init__(
            source_train_marginals=source_train_marginals,
            marginal_estimator_name=(
                f"{marginal_estimator_name}"
                f"_{underlying_estimator.marginal_estimator_name}"
            ),
        )
        self.marginal_est = source_train_marginals
        self.underlying_estimator = underlying_estimator

    def get_marginal_estimate(self, dataloader, use_current_marginal_estimate):
        if not use_current_marginal_estimate:
            cur_marginal_est = copy.deepcopy(self.marginal_est)
            self.update_marginal_estimate(dataloader)
        else:
            self.update_marginal_estimate(dataloader)
            cur_marginal_est = copy.deepcopy(self.marginal_est)
        return cur_marginal_est

    def update_marginal_estimate(self, dataloader):
        raise NotImplementedError


class FollowTheHistoryEstimator(ShiftEstimatorWithMemory):
    def __init__(
        self,
        source_train_marginals,
        underlying_estimator: LocalShiftEstimator,
        marginal_estimator_name="FTH",
    ) -> None:
        super().__init__(
            source_train_marginals=source_train_marginals,
            underlying_estimator=underlying_estimator,
            marginal_estimator_name=marginal_estimator_name,
        )
        self.marginal_est_history = None

    def update_marginal_estimate(self, dataloader):
        cur_marginal_est = self.underlying_estimator.get_marginal_estimate(dataloader)
        if self.marginal_est_history is None:
            self.marginal_est_history = mscu.np2torch(cur_marginal_est).unsqueeze(0)
        else:
            self.marginal_est_history = torch.vstack(
                (self.marginal_est_history, mscu.np2torch(cur_marginal_est))
            )

        self.marginal_est = torch.mean(self.marginal_est_history, dim=0)


class FollowTheFixedWindowHistoryEstimator(FollowTheHistoryEstimator):
    def __init__(
        self,
        source_train_marginals,
        underlying_estimator: LocalShiftEstimator,
        window_size,
    ) -> None:
        super().__init__(
            source_train_marginals=source_train_marginals,
            underlying_estimator=underlying_estimator,
            marginal_estimator_name="FTFWH",
        )
        self.num_records = 0
        self.window_size = window_size

    def update_marginal_estimate(self, dataloader):
        if self.num_records >= int(self.window_size):
            self.marginal_est_history = self.marginal_est_history[1:, :]
        super().update_marginal_estimate(dataloader)
        self.num_records += 1


class FollowLeadingHistoryFollowTheLeaderEstimator(ShiftEstimatorWithMemory):
    def __init__(
        self,
        source_train_marginals,
        underlying_estimator: LocalShiftEstimator,
        num_labels,
        meta_lr=None,
    ) -> None:
        super().__init__(
            source_train_marginals,
            underlying_estimator,
            marginal_estimator_name="FLH-FTL",
        )
        self.num_labels = num_labels
        self.meta_lr = meta_lr
        self.compute_meta_lr()

        self.marginal_est_history = None
        self.current_time = 0

        ## FLH parameters
        # Leftmost experts track only the immediate marginal history
        self.experts = np.zeros((self.num_labels, 0))  # num_labels x num_experts
        self.experts_coeff = np.zeros(1)

    def compute_meta_lr(self):
        if self.meta_lr is None:
            self.meta_lr = 1 / self.num_labels
        logger.info(
            f"{self.marginal_estimator_name}'s meta learning rate: {self.meta_lr}"
        )

    def update_marginal_estimate(self, dataloader):
        """Update target marginal estimate based on predicted labels"""
        # Set current reweight
        num_records = (
            self.marginal_est_history.shape[1]
            if self.marginal_est_history is not None
            else 0
        )
        if num_records > 0:
            self.marginal_est = self.experts @ self.experts_coeff

        # Compute current marginal estimate
        cur_marginal_est = self.underlying_estimator.get_marginal_estimate(dataloader)
        if self.marginal_est_history is None:
            self.marginal_est_history = np.expand_dims(cur_marginal_est, 1)
        else:
            self.marginal_est_history = np.hstack(
                (self.marginal_est_history, np.expand_dims(cur_marginal_est, 1))
            )

        # Update expert coefficient
        for expert_idx in range(num_records - 1):
            decay = np.exp(
                -self.meta_lr
                * np.linalg.norm(self.experts[:, expert_idx] - cur_marginal_est) ** 2
            )
            self.experts_coeff[expert_idx] = self.experts_coeff[expert_idx] * decay

        if num_records > 0:
            # Add new expert at the end
            self.experts_coeff = self.experts_coeff / np.sum(self.experts_coeff)
            self.experts_coeff = self.experts_coeff * num_records / (num_records + 1)
            self.experts_coeff = np.append(self.experts_coeff, 1 / (num_records + 1))
        else:
            self.experts_coeff = np.ones(1)

        # Update experts. Experts on the left track the most recent few.
        self.experts = np.concatenate(
            (np.expand_dims(cur_marginal_est, 1), self.experts), axis=1
        )
        for expert_idx in range(1, num_records + 1):
            self.experts[:, expert_idx] = np.mean(
                self.marginal_est_history[:, -(expert_idx + 1) :], axis=1
            )


class RegularOnlineGradientEstimator(ShiftEstimatorWithMemory):
    def __init__(
        self,
        source_train_marginals,
        underlying_estimator: LocalShiftEstimator,
        ref_labels,
        ref_prob,
        num_labels,
        lr=None,
        lipschitz=None,
        marginal_estimator_name="ROGD",
        use_smooth_grad=True,
    ) -> None:
        super().__init__(
            source_train_marginals=source_train_marginals,
            underlying_estimator=underlying_estimator,
            marginal_estimator_name=marginal_estimator_name,
        )
        self.ref_labels = ref_labels
        self.ref_prob = mscu.torch2np(ref_prob)
        self.num_labels = num_labels
        self.ref_marg = lsu.get_label_marginals(ref_labels, self.num_labels)

        # Learning rate parameters
        self.lipschitz = lipschitz
        self.lr = lr

        self.use_smooth_grad = use_smooth_grad

    def get_zero_one_loss(self, p, q):
        pred = lsu.get_reweighted_predictions(self.ref_prob, self.ref_marg, p)
        conf_matrix = lsu.get_confusion_matrix(self.ref_labels, pred, self.num_labels)
        error_rates_per_labels = 1 - np.diag(conf_matrix)

        return error_rates_per_labels @ q

    def get_zero_one_grad(self, p, q):
        loss_p = lambda p_0: self.get_zero_one_loss(p_0, q)
        grad = mscu.finite_gradient(loss_p, x0=p, delta=1e-2, k=3)
        return grad

    def get_smooth_loss(self, p, q):
        weighted_prob = np.einsum("ij,j->ij", self.ref_prob, p)
        weighted_prob = weighted_prob / weighted_prob.sum(axis=1)[:, np.newaxis]
        conf_matrix = lsu.get_soft_confusion_matrix(
            self.ref_labels, weighted_prob, self.num_labels
        )
        error_rates_per_labels = 1 - np.diag(conf_matrix)
        return error_rates_per_labels @ q

    def get_smooth_grad(self, p, q):
        loss_p = lambda p_0: self.get_smooth_loss(p_0, q)
        grad = mscu.finite_gradient(loss_p, x0=p, delta=1e-2, k=3)
        return grad

    def get_analytical_smooth_grad(self, p, q):
        num_labels = p.shape[0]
        num_data = self.ref_prob.shape[0]

        # Build d(diag_j)/dp_i
        grad_matrix = np.zeros((num_labels, num_labels))
        label_idcs = [np.where(self.ref_labels == j)[0] for j in range(num_labels)]

        weighted_prob = np.einsum("ij,j->ij", self.ref_prob, p)
        for i in range(num_labels):
            for j in range(num_labels):
                idx_label_j = label_idcs[j]
                if p[j] < 1e-7:
                    continue

                grad = (1 / num_data) * np.sum(
                    -self.ref_prob[idx_label_j, i]
                    * weighted_prob[idx_label_j, j]
                    / (np.sum(weighted_prob[idx_label_j, :], axis=1)) ** 2
                )
                if i == j:
                    new_term = (1 / num_data) * np.sum(
                        self.ref_prob[idx_label_j, i]
                        / (np.sum(weighted_prob[idx_label_j, :], axis=1))
                    )
                    grad += new_term
                grad_matrix[i, j] = grad

        return (-grad_matrix) @ q

    def estimate_lipschitz(self, ntries=30):
        """Estimate lipchitz constant by sampling over uniform p and all one-hot vectors
        of predicted label
        """
        lipschitz = -1
        iterator = tqdm(range(ntries), desc="Finding lipschitz constant")
        for _ in iterator:
            p = lsu.get_random_probability(self.num_labels)
            for label_idx in range(self.num_labels):
                e_label = np.zeros(self.num_labels)
                e_label[label_idx] = 1.0
                q_hat = lsu.BBSE(
                    self.underlying_estimator.confusion_matrix,
                    source_train_marginals=self.source_train_marginals,
                    y_pred_marginals=e_label,
                )
                grad = self.get_zero_one_grad(p, q_hat)
                cur_lipschitz = np.linalg.norm(grad)
                if cur_lipschitz > lipschitz:
                    lipschitz = cur_lipschitz
                    logger.debug(
                        f"{self.marginal_estimator_name} New lipschitz {lipschitz}"
                    )
        self.lipschitz = lipschitz

        logger.info(
            f"{self.marginal_estimator_name} lipschitz constant found: {lipschitz}"
        )

    def estimate_learning_rate(self, total_time, lipschitz_ntries, seed=None):
        """Computing default lr according ROGD's authors
        see Theorem 2, pg.5 of https://arxiv.org/pdf/2107.04520.pdf
        """
        if self.lr is not None:
            logger.info(f"{self.marginal_estimator_name} lr is already set to {self.lr}")
            return

        if self.lipschitz is None:
            if seed is not None:
                mscu.set_seed(seed)
            self.estimate_lipschitz(ntries=lipschitz_ntries)
        self.lr = (2 / total_time) ** (1 / 2) / self.lipschitz

        logger.info(f"{self.marginal_estimator_name} Learning rate found: {self.lr}")

    def update_marginal_estimate(self, dataloader):
        local_marg_est = self.underlying_estimator.get_marginal_estimate(dataloader)
        if self.use_smooth_grad:
            marg_est_grad = self.get_analytical_smooth_grad(
                self.marginal_est, local_marg_est
            )
            # Avoid numerical instability
            if np.linalg.norm(marg_est_grad) > 10:
                marg_est_grad = self.get_smooth_grad(self.marginal_est, local_marg_est)
        else:
            marg_est_grad = self.get_zero_one_grad(self.marginal_est, local_marg_est)

        self.marginal_est = self.marginal_est - self.lr * marg_est_grad
        self.marginal_est = mscu.proj_to_probsimplex(self.marginal_est)

        logger.debug(
            f"{self.marginal_estimator_name} Local marginal estimate: {local_marg_est}"
        )
        logger.debug(f"{self.marginal_estimator_name} grad: {marg_est_grad}")
        logger.debug(
            f"{self.marginal_estimator_name} marginal estimate: {self.marginal_est}"
        )


class SimpleLocalShiftEstimator(ShiftEstimatorWithMemory):
    def __init__(
        self,
        source_train_marginals,
        underlying_estimator: LocalShiftEstimator,
    ) -> None:
        super().__init__(
            source_train_marginals=source_train_marginals,
            underlying_estimator=underlying_estimator,
            marginal_estimator_name="SIMP-LOCAL",
        )

    def update_marginal_estimate(self, dataloader):
        self.marginal_est = self.underlying_estimator.get_marginal_estimate(dataloader)


class OracleEstimator(ShiftEstimatorWithMemory):
    def __init__(
        self,
        source_train_marginals,
        underlying_estimator: LocalShiftEstimator,
        num_labels,
    ) -> None:
        self.num_labels = num_labels
        super().__init__(
            source_train_marginals=source_train_marginals,
            underlying_estimator=underlying_estimator,  # This is strictly not needed
            marginal_estimator_name="ORACLE",
        )

    def get_marginal_estimate(self, dataloader, use_current_marginal_estimate):
        """Oracle always use current estimate"""
        self.update_marginal_estimate(dataloader)
        cur_marginal_est = copy.deepcopy(self.marginal_est)
        return cur_marginal_est

    def update_marginal_estimate(self, dataloader):
        self.marginal_est = lsu.get_label_marginals(
            dataloader.dataset.y_array, num_labels=self.num_labels
        )


class BlackBoxShiftEstimator(LocalShiftEstimator):
    def __init__(
        self,
        model,
        source_train_marginals,
        num_labels,
        ref_dataloader,
        confusion_matrix=None,
    ) -> None:
        super().__init__(
            model=model,
            source_train_marginals=source_train_marginals,
            num_labels=num_labels,
            ref_dataloader=ref_dataloader,
            marginal_estimator_name="BBSE",
            confusion_matrix=confusion_matrix,
        )

    def _get_marginal_estimate(self, pred_marginals):
        marginal_est = lsu.BBSE(
            confusion_matrix=self.confusion_matrix,
            source_train_marginals=self.source_train_marginals,
            y_pred_marginals=pred_marginals,
        )
        return marginal_est


class RegularizedShiftEstimator(LocalShiftEstimator):
    def __init__(
        self,
        model,
        source_train_marginals,
        source_train_num_data,
        num_labels,
        ref_dataloader,
        confusion_matrix=None,
    ) -> None:
        super().__init__(
            model=model,
            source_train_marginals=source_train_marginals,
            num_labels=num_labels,
            ref_dataloader=ref_dataloader,
            marginal_estimator_name="RLLS",
            confusion_matrix=confusion_matrix,
        )
        self.source_train_num_data = source_train_num_data
        self.ref_marginals = lsu.get_label_marginals(
            ref_dataloader.dataset.y_array, self.num_labels
        )

    def _get_marginal_estimate(self, pred_marginals):
        marginal_est = lsu.RLLS(
            confusion_matrix=self.confusion_matrix,
            mu_y=pred_marginals,
            mu_train_y=self.source_train_marginals,
            num_labels=self.num_labels,
            n_train=self.source_train_num_data,
        )
        return marginal_est


class MaximumLikelihoodShiftEstimator(LocalShiftEstimator):
    def __init__(
        self,
        model,
        source_soft_marginals,
        num_labels,
    ) -> None:
        self.source_soft_marginals = source_soft_marginals
        super().__init__(
            model=model,
            source_train_marginals=None,
            num_labels=num_labels,
            ref_dataloader=None,
            marginal_estimator_name="MLLS",
            confusion_matrix=(None,),
        )

    def get_marginal_estimate(self, dataloader):
        y_prob = self.model.get_predicted_probabilities(dataloader=dataloader)
        return self._get_marginal_estimate(y_prob.cpu().numpy())

    def _get_marginal_estimate(self, pred_prob):
        # TODO: Does MLLS take soft marginals from source train or source test
        marginal_est = lsu.MLLS(
            source_pred_soft_marginals=self.source_soft_marginals,
            test_pred_prob=pred_prob,
            numClasses=self.num_labels,
        )
        return marginal_est


SUPPORTED_LOCAL_SHIFT_ESTIMATORS = ["bbse", "mlls", "rlls"]


### Online Classifiers
class UnsupervisedReweightingClassifier:
    def __init__(
        self,
        model: BaseModel,
        marginal_estimator: ShiftEstimatorWithMemory,
        source_train_marginals,
        num_labels,
        use_current_marginal_estimate,
    ) -> None:
        self.marginal_estimator = marginal_estimator
        self.model = model
        if isinstance(self.model, BaseModelNN):
            self.model.eval()
        self.source_train_marginals = source_train_marginals
        self.num_labels = num_labels
        self.use_current_marginal_estimate = use_current_marginal_estimate
        self.name = f"RW_{self.marginal_estimator.marginal_estimator_name}"

    def predict(self, dataloader, marginal_estimate):
        """Make predictions"""
        y_true = np.zeros(0)
        y_pred = np.zeros(0)

        # Make prediction
        if isinstance(self.model, BaseModelNN):
            self.model.eval()
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(prju.Parameters.device), y.to(prju.Parameters.device)
                logits = self.model.forward(X)
                cur_prob = nn.Softmax(dim=1)(logits)
                cur_pred = lsu.get_reweighted_predictions(
                    cur_prob.cpu().numpy(), self.source_train_marginals, marginal_estimate
                )
                y_true = np.concatenate((y_true, y.flatten().cpu().numpy()))
                y_pred = np.concatenate((y_pred, cur_pred))
        return y_pred, y_true

    def predict_online_label_shift(self, dataloaders):
        results = {
            "true_labels": [],
            "pred_labels": [],
            "true_label_marginals": [],
            "est_label_marginals": [],
            "accuracies_per_time": [],
            "l2-sq_marginal_errors": [],
            "accuracy": None,
        }
        if isinstance(self.model, BaseModelNN):
            self.model.eval()
        with torch.no_grad():
            iterator = tqdm(dataloaders, desc=f"Making {self.name} prediction")
            for t, dataloader in enumerate(iterator):
                marg_est = self.marginal_estimator.get_marginal_estimate(
                    dataloader, self.use_current_marginal_estimate
                )
                cur_y_pred, cur_y_true = self.predict(
                    dataloader=dataloader, marginal_estimate=marg_est
                )
                results["true_labels"].append(cur_y_true)
                results["pred_labels"].append(cur_y_pred)
                cur_accuracy = lsu.get_accuracy(cur_y_true, cur_y_pred)
                results["accuracies_per_time"].append(cur_accuracy)
                cur_true_marginals = lsu.get_label_marginals(cur_y_true, self.num_labels)

                results["true_label_marginals"].append(cur_true_marginals)
                cur_est_marginals = mscu.torch2np(marg_est)
                results["est_label_marginals"].append(cur_est_marginals)
                cur_l2sq_marg_error = np.sum(
                    (cur_true_marginals - cur_est_marginals) ** 2
                )
                results["l2-sq_marginal_errors"].append(cur_l2sq_marg_error)

                logger.debug("*" * 84)
                logger.debug(f"Time={t}")
                logger.debug(f"Current Marginal Estimate {cur_est_marginals}")
                logger.debug(f"Current True Label Marg {cur_true_marginals}")
                logger.debug(
                    f"Current Pred Label Marg {lsu.get_label_marginals(cur_y_pred, self.num_labels)}"
                )
                logger.debug(f"Current Accuracy {cur_accuracy}")
                logger.debug(f"Current L2-sq marginal error {cur_l2sq_marg_error}")

        accuracy = lsu.get_accuracy(
            np.concatenate(results["true_labels"]), np.concatenate(results["pred_labels"])
        )
        results["accuracy"] = accuracy.numpy()
        return results


class UnbiasedOnlineGradientDescentClassifier:
    def __init__(
        self,
        model,
        ref_dataloader,
        num_labels,
        marginal_estimator: MarginalEstimator,
        weight_name,
        source_train_marginals,
        use_current_marginal_estimate,
        lr=None,
        model_param_norms=None,
        clf_name="UOGD",
    ) -> None:
        self.device = prju.Parameters.device
        self.model: BaseModelWithLinearWeight = model.to(self.device)

        self.ref_dataloader = ref_dataloader

        self.num_labels = num_labels
        self.model_param_norms = model_param_norms
        self.lr = lr
        self.marginal_estimator: MarginalEstimator = marginal_estimator
        self.weight_name = weight_name
        self.source_train_marginals = source_train_marginals
        self.use_current_marginal_estimate = use_current_marginal_estimate
        self.compute_model_param_norms()

        self.clf_name = clf_name
        self.name = f"{clf_name}_{self.marginal_estimator.marginal_estimator_name}"

    def compute_lr_from_total_time(self, total_time):
        """Computing default lr according to UOGD's authors
        see pg.16 of https://arxiv.org/pdf/2207.02121.pdf
        """
        if self.lr is None:
            model_norms = list(self.model_param_norms.values())
            param_norm = torch.zeros(size=(1,), device=model_norms[0].device)
            for model_norm in model_norms:
                param_norm += model_norm
            self.lr = (
                2
                * param_norm
                / self.model.weight_lipschitz_estimate
                / (total_time ** (1 / 2))
            )
            logger.info(f"{self.name} learning rate: {self.lr}")

    def compute_model_param_norms(self):
        if self.model_param_norms is None:
            self.model_param_norms = {}
            for name, param in self.model.state_dict().items():
                if name == self.weight_name:
                    self.model_param_norms[name] = torch.linalg.norm(param.data)
                    logger.debug(f"Param {name} with norm {self.model_param_norms[name]}")

    def project_model_parameters(self):
        """Project base model parameters to saved parameter norms"""
        for name, param in self.model.state_dict().items():
            if name == self.weight_name:
                new_param = (
                    param / (torch.linalg.norm(param.data)) * self.model_param_norms[name]
                )
                param.copy_(new_param)

    def update_weight(self, q_hat, ref_features, ref_y):
        """Update last linear layer based on predicted label marginals"""
        # Train only last layer
        self.model.feature_extractor.eval()
        self.model.linear.train()

        # Take weighted gradient step based on source loss
        loss_weight = (
            torch.from_numpy(q_hat / self.source_train_marginals).float().to(self.device)
        )
        criterion = nn.CrossEntropyLoss(weight=loss_weight)
        total_loss = 0.0
        optimizer = torch.optim.SGD(
            # self.model.linear.weight,
            self.model.linear.parameters(),
            lr=self.lr.item(),
            weight_decay=0.0,
            momentum=0.0,
        )

        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            loss = criterion(self.model.linear(ref_features), ref_y)
            loss.backward()
            total_loss = loss.item()
            optimizer.step()

            self.project_model_parameters()

        return total_loss

    def predict(self, dataloader):
        """Make predictions"""
        y_pred, y_true = self.model.get_predictions(dataloader)
        return y_pred, y_true

    def predict_online_label_shift(self, dataloaders):
        def _predict_subroutine(results, dataloader):
            cur_y_pred, cur_y_true = self.predict(dataloader)
            results["true_labels"].append(cur_y_true.numpy())
            results["pred_labels"].append(cur_y_pred.numpy())
            results["accuracies_per_time"].append(
                lsu.get_accuracy(cur_y_true, cur_y_pred).numpy()
            )
            results["true_label_marginals"].append(
                lsu.get_label_marginals(cur_y_true, self.num_labels)
            )

        def _update_subroutine(results, dataloader, ref_features, ref_y):
            q_hat = self.marginal_estimator.get_marginal_estimate(dataloader)
            results["est_label_marginals"].append(q_hat)

            logger.debug(f"{self.name} estimate of current labels marginals: {q_hat}")
            self.update_weight(q_hat, ref_features, ref_y)

        # Precompute ref datas features.
        self.model.feature_extractor.eval()
        ref_features = torch.zeros((0, self.model.feature_dim), device=self.device)
        ref_y = torch.zeros(0, device=self.device, dtype=torch.int64)
        with torch.no_grad():
            for X, y in self.ref_dataloader:
                X, y = X.to(self.device), y.to(self.device)

                cur_feats = self.model.feature_extractor(X)
                ref_features = torch.cat((ref_features, cur_feats), dim=0)
                ref_y = torch.cat((ref_y, y))

        results = {
            "true_labels": [],
            "pred_labels": [],
            "true_label_marginals": [],
            "est_label_marginals": [],
            "accuracies_per_time": [],
            "accuracy": None,
        }

        with torch.no_grad():
            iterator = tqdm(dataloaders, desc=f"Making {self.name} prediction")
            for dataloader in iterator:
                if not self.use_current_marginal_estimate:
                    _predict_subroutine(results, dataloader)
                    _update_subroutine(results, dataloader, ref_features, ref_y)
                else:
                    _update_subroutine(results, dataloader, ref_features, ref_y)
                    _predict_subroutine(results, dataloader)

        accuracy = lsu.get_accuracy(
            np.concatenate(results["true_labels"]), np.concatenate(results["pred_labels"])
        )
        results["accuracy"] = accuracy.numpy()
        return results


class ATLAS:
    clf_name = "ATLAS"

    def __init__(
        self,
        model,
        ref_dataloader,
        num_labels,
        marginal_estimator: MarginalEstimator,
        weight_name,
        total_time,
        source_train_marginals,
        use_current_marginal_estimate,
        base_lr_pool=None,
        model_param_norms=None,
        meta_lr=None,
    ) -> None:
        self.model: BaseModelWithLinearWeight = model
        self.ref_dataloader = ref_dataloader
        self.device = self.model.device
        self.total_time = total_time
        self.source_train_marginals = source_train_marginals
        self.marginal_estimator: MarginalEstimator = marginal_estimator
        self.num_labels = num_labels
        self.weight_name = weight_name
        self.model_param_norms = model_param_norms
        self.use_current_marginal_estimate = (use_current_marginal_estimate,)
        self.compute_model_param_norms()
        self.base_lr_pool = base_lr_pool
        self.name = f"{self.clf_name}_{self.marginal_estimator.marginal_estimator_name}"
        self.compute_lr_pool()

        self.num_learners = len(self.base_lr_pool)
        self.base_leaners = [
            UnbiasedOnlineGradientDescentClassifier(
                model=copy.deepcopy(model),
                ref_dataloader=ref_dataloader,
                num_labels=num_labels,
                marginal_estimator=marginal_estimator,
                weight_name=weight_name,
                source_train_marginals=source_train_marginals,
                use_current_marginal_estimate=use_current_marginal_estimate,
                lr=lr,
                model_param_norms=copy.deepcopy(self.model_param_norms),
            )
            for lr in self.base_lr_pool
        ]
        self.base_learner_weights = (
            torch.ones(self.num_learners, device=self.device) / self.num_learners
        )
        self.base_learner_accm_risks = torch.zeros(
            size=(self.num_learners,), device=self.device
        )
        self.meta_lr = meta_lr
        self.compute_meta_lr()

    def compute_model_param_norms(self):
        if self.model_param_norms is not None:
            return

        self.model_param_norms = {}
        for name, param in self.model.state_dict().items():
            if name == self.weight_name:
                self.model_param_norms[name] = torch.linalg.norm(param.data)

    def compute_lr_pool(self):
        """Computing default lr pool according to ATLAS's authors
        see Theorem 2, pg.7 of https://arxiv.org/pdf/2207.02121.pdf
        """
        if self.base_lr_pool is not None:
            return

        self.num_learners = int(1 + np.ceil(np.log2(1 + 2 * self.total_time) / 2))

        # Calculate minimum learning rate as specified by the paper
        model_norms = list(self.model_param_norms.values())
        param_norm = torch.zeros(size=(1,), device=model_norms[0].device)
        for model_norm in model_norms:
            param_norm += model_norm
        eigvals = np.real(np.linalg.eigvals(self.marginal_estimator.confusion_matrix))
        sigma = np.min(eigvals)
        base_lr_rate = (
            2
            * param_norm.cpu()
            * sigma
            / (
                2
                * self.model.weight_lipschitz_estimate
                * (self.num_labels * self.total_time) ** (1 / 2)
            )
        )

        self.base_lr_pool = []
        for i in range(self.num_learners):
            lr = torch.tensor(base_lr_rate * (2**i), device=self.device)
            self.base_lr_pool.append(lr)
        logger.info(f"{self.name} number of base learners: {self.num_learners}")
        logger.info(f"{self.name} learning rate pool: {self.base_lr_pool}")

    def compute_meta_lr(self):
        """Computing default meta lr according to ATLAS's authors
        see pg.6 of https://arxiv.org/pdf/2207.02121.pdf
        """
        if self.meta_lr is not None:
            return
        self.meta_lr = 2 * (np.log(self.num_learners) / self.total_time) ** (1 / 2)
        logger.info(f"{self.name} meta learning rate: {self.meta_lr}")

    def combine_weights(self):
        for name, param in self.model.named_parameters():
            if name == self.weight_name:
                combined_param = torch.zeros(size=param.shape, device=self.device)
                for idx, learner in enumerate(self.base_leaners):
                    base_weight = learner.model.state_dict()[name]
                    combined_param += base_weight * self.base_learner_weights[idx]
                with torch.no_grad():
                    param.copy_(combined_param)

    def predict(self, dataloader):
        """Make predictions"""
        y_pred, y_true = self.model.get_predictions(dataloader)
        return y_pred, y_true

    def predict_online_label_shift(self, dataloaders):
        def _predict_subroutine(results, dataloader):
            self.combine_weights()
            cur_y_pred, cur_y_true = self.predict(dataloader)
            results["true_labels"].append(cur_y_true.numpy())
            results["pred_labels"].append(cur_y_pred.numpy())
            results["accuracies_per_time"].append(
                lsu.get_accuracy(cur_y_true, cur_y_pred).numpy()
            )
            results["true_label_marginals"].append(
                lsu.get_label_marginals(cur_y_true, self.num_labels)
            )

        def _update_subroutine(results, dataloader, ref_features, ref_y):
            q_hat = self.marginal_estimator.get_marginal_estimate(dataloader)
            for idx, learner in enumerate(self.base_leaners):
                self.base_learner_accm_risks[idx] += learner.update_weight(
                    q_hat, ref_features, ref_y
                )
            self.base_learner_weights = mscu.softmax(
                -self.meta_lr * self.base_learner_accm_risks
            )

            results["est_label_marginals"].append(q_hat)
            results["base_learner_weights"].append(
                self.base_learner_weights.cpu().numpy()
            )

        # Precompute ref datas features.
        self.model.feature_extractor.eval()
        ref_features = torch.zeros((0, self.model.feature_dim), device=self.device)
        ref_y = torch.zeros(0, device=self.device, dtype=torch.int64)
        with torch.no_grad():
            for X, y in self.ref_dataloader:
                X, y = X.to(self.device), y.to(self.device)

                cur_feats = self.model.feature_extractor(X)
                ref_features = torch.cat((ref_features, cur_feats), dim=0)
                ref_y = torch.cat((ref_y, y))

        results = {
            "true_labels": [],
            "pred_labels": [],
            "true_label_marginals": [],
            "est_label_marginals": [],
            "accuracies_per_time": [],
            "base_learner_weights": [],
            "accuracy": None,
        }
        with torch.no_grad():
            iterator = tqdm(dataloaders, desc=f"Making {self.name} prediction")
            for dataloader in iterator:
                if not self.use_current_marginal_estimate:
                    _predict_subroutine(results, dataloader)
                    _update_subroutine(results, dataloader, ref_features, ref_y)
                else:
                    _update_subroutine(results, dataloader, ref_features, ref_y)
                    _predict_subroutine(results, dataloader)

        accuracy = lsu.get_accuracy(
            np.concatenate(results["true_labels"]), np.concatenate(results["pred_labels"])
        )
        results["accuracy"] = accuracy.numpy()
        return results


class LinearLayerRetrainedClassifier:
    def __init__(
        self,
        model,
        ref_dataloader,  # The data used for retraining linear layer
        marginal_estimator: ShiftEstimatorWithMemory,
        num_labels,
        source_train_marginals,
        use_current_marginal_estimate,
        lr=1e-1,
        epoch=200,
        clf_name="LIN-RETRAIN",
    ) -> None:
        self.device = prju.Parameters.device
        self.model: BaseModelWithLinearWeight = model.to(self.device)

        self.ref_dataloader = ref_dataloader

        self.num_labels = num_labels
        self.lr = lr
        self.epoch = epoch
        self.marginal_estimator: MarginalEstimator = marginal_estimator
        self.source_train_marginals = source_train_marginals
        self.use_current_marginal_estimate = use_current_marginal_estimate

        self.clf_name = clf_name
        self.name = f"{clf_name}_{self.marginal_estimator.marginal_estimator_name}"

    def update_weight(self, q_hat, ref_features, ref_y):
        """Update last linear layer based on predicted label marginals"""
        # Train only last layer
        self.model.feature_extractor.eval()
        self.model.linear = nn.Linear(
            self.model.linear.in_features,
            self.model.linear.out_features,
            bias=False,
            device=self.device,
        )
        self.model.linear.train()

        # Take weighted gradient step based on source loss
        loss_weight = (
            torch.from_numpy(q_hat / self.source_train_marginals).float().to(self.device)
        )
        criterion = nn.CrossEntropyLoss(weight=loss_weight)
        optimizer = torch.optim.SGD(
            self.model.linear.parameters(),
            lr=self.lr,
            weight_decay=0.0,
            momentum=0.5,
        )

        with torch.set_grad_enabled(True):
            for cur_epoch in range(self.epoch):
                optimizer.zero_grad()
                loss = criterion(self.model.linear(ref_features), ref_y)
                loss.backward()
                optimizer.step()
                if cur_epoch % 10 == 0:
                    logger.debug(f"{self.name} epoch {cur_epoch} loss {loss}")
        self.model.linear.eval()

    def predict(self, dataloader):
        """Make predictions"""
        y_pred, y_true = self.model.get_predictions(dataloader)
        return y_pred, y_true

    def predict_online_label_shift(self, dataloaders):
        def _predict_subroutine(results, dataloader):
            cur_y_pred, cur_y_true = self.predict(dataloader)
            results["true_labels"].append(cur_y_true.numpy())
            results["pred_labels"].append(cur_y_pred.numpy())
            results["accuracies_per_time"].append(
                lsu.get_accuracy(cur_y_true, cur_y_pred).numpy()
            )
            results["true_label_marginals"].append(
                lsu.get_label_marginals(cur_y_true, self.num_labels)
            )

        def _update_subroutine(results, dataloader, ref_features, ref_y):
            q_hat = self.marginal_estimator.get_marginal_estimate(
                dataloader, self.use_current_marginal_estimate
            )
            results["est_label_marginals"].append(q_hat)

            logger.debug(f"{self.name} estimate of current labels marginals: {q_hat}")
            self.update_weight(q_hat, ref_features, ref_y)

        # Precompute ref datas features.
        self.model.feature_extractor.eval()
        ref_features = torch.zeros((0, self.model.feature_dim), device=self.device)
        ref_y = torch.zeros(0, device=self.device, dtype=torch.int64)
        with torch.no_grad():
            for X, y in self.ref_dataloader:
                X, y = X.to(self.device), y.to(self.device)

                cur_feats = self.model.feature_extractor(X)
                ref_features = torch.cat((ref_features, cur_feats), dim=0)
                ref_y = torch.cat((ref_y, y))

        results = {
            "true_labels": [],
            "pred_labels": [],
            "true_label_marginals": [],
            "est_label_marginals": [],
            "accuracies_per_time": [],
            "accuracy": None,
        }

        self.model.eval()
        with torch.no_grad():
            iterator = tqdm(dataloaders, desc=f"Making {self.name} prediction")
            for dataloader in iterator:
                if not self.use_current_marginal_estimate:
                    _predict_subroutine(results, dataloader)
                    _update_subroutine(results, dataloader, ref_features, ref_y)
                else:
                    _update_subroutine(results, dataloader, ref_features, ref_y)
                    _predict_subroutine(results, dataloader)

        accuracy = lsu.get_accuracy(
            np.concatenate(results["true_labels"]), np.concatenate(results["pred_labels"])
        )
        results["accuracy"] = accuracy.numpy()
        return results


class BNAdaptClassifier:
    def __init__(self, model) -> None:
        self.model = model
        self.name = "BNAdapt"

        self.device = self.model.device

    def predict_online_label_shift(self, dataloaders):
        results = {
            "true_labels": [],
            "pred_labels": [],
            "accuracies_per_time": [],
            "accuracy": None,
        }

        with torch.no_grad():
            iterator = tqdm(dataloaders, desc=f"Making {self.name} prediction")
            for dataloader in iterator:
                # BN adapt
                self.model.train()
                for X, y in dataloader:
                    X, y = X.to(self.device), y.to(self.device)
                    self.model.forward(X)

                self.model.eval()
                cur_y_pred, cur_y_true = self.model.get_predictions(dataloader)
                results["true_labels"].append(cur_y_true)
                results["pred_labels"].append(cur_y_pred)
                cur_accuracy = lsu.get_accuracy(cur_y_true, cur_y_pred)
                results["accuracies_per_time"].append(cur_accuracy)

        accuracy = lsu.get_accuracy(
            np.concatenate(results["true_labels"]), np.concatenate(results["pred_labels"])
        )
        results["accuracy"] = accuracy.numpy()
        return results
