import copy
import logging
import math
import time
from datetime import datetime

import numpy as np
import torch


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def np2torch(array):
    if type(array) != torch.Tensor:
        return torch.from_numpy(array)
    return array


def torch2np(array):
    if type(array) != np.ndarray:
        return array.numpy()
    return array


def softmax(x, axis=None):
    if type(x) is torch.Tensor:
        x = x - x.max()
        y = torch.exp(x)
        return y / y.sum()
    else:
        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=axis, keepdims=True)


def prob_to_logit(X):
    """
    Predict logarithm of probability estimates.
    The returned estimates for all classes are ordered by the
    label of classes.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Vector to be scored, where `n_samples` is the number of samples and
        `n_features` is the number of features.
    Returns
    -------
    T : array-like of shape (n_samples, n_classes)
        Returns the log-probability of the sample for each class in the
        model, where classes are ordered as they are in ``self.classes_``.
    """
    return np.log(X + 1e-10)


def proj_to_probsimplex(vector):
    """Implementation taken from https://arxiv.org/pdf/1101.6081.pdf"""
    n = vector.shape[0]
    u = np.flip(np.sort(vector))
    for j in reversed(range(n)):
        if u[j] + (1 / (j + 1)) * (1 - np.sum(u[: j + 1])) > 0:
            p = j
            break
    lmda = 1 / (p + 1) * (1 - np.sum(u[: p + 1]))
    simplex = vector + lmda
    simplex[simplex < 0] = 0

    return simplex


def pytorch2sklearn(dataloader):
    dim = dataloader.dataset.__getitem__(0)[0].flatten().shape[0]
    all_X, all_y = np.zeros((0, dim)), np.zeros(0)
    for X, y in dataloader:
        X, y = torch.flatten(X, start_dim=1).numpy(), y.numpy()
        all_X = np.concatenate((all_X, X), axis=0)
        all_y = np.concatenate((all_y, y), axis=0)
    return all_X, all_y


def finite_gradient(function, x0, delta, k):
    """Taken from https://arxiv.org/abs/2107.04520"""
    num_params = x0.shape[0]
    grad = np.zeros(num_params)

    for i in range(num_params):
        e_i = np.zeros(num_params)
        e_i[i] = 1
        for j in range(1, k + 1):
            function_diff = function(x0 + j * delta * e_i) - function(
                x0 - j * delta * e_i
            )
            alpha_j = 2 * (-1) ** (j + 1) * math.comb(k, k - j) / math.comb(k + j, k)
            grad[i] += function_diff * alpha_j / (2 * delta * j)
    return grad


def get_logger(level=None, file_handler=None):
    logging.basicConfig(format="%(asctime)s %(message)s")
    logger = logging.getLogger("online_label_shift")
    if file_handler is not None:
        formatter = logging.Formatter("%(asctime)s %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if level is not None:
        logger.setLevel(level)
    return logger


def get_current_time():
    now = datetime.now()
    return now.strftime("%Y-%m-%d-%H:%M:%S")


def serialize_dict(d):
    sd = copy.deepcopy(d)
    for key, val in d.items():
        if type(val) is dict:
            sd[key] = serialize_dict(val)
        if type(val) is np.ndarray or torch.is_tensor(val):
            sd[key] = str(val)
    return sd


def tick():
    return time.time()


def tock(start_time, unit="m"):
    diff = time.time() - start_time

    ratio = {"s": 1, "m": 60, "h": 3600}
    return diff / ratio[unit]


def boolean(input):
    true_conditions = [
        lambda x: x.lower() == "true",
        lambda x: x.lower() == "yes",
        lambda x: x == "1",
    ]
    for cond in true_conditions:
        if cond(input) is True:
            return True
    return False


def acc2error(acc):
    return 100 * (1 - acc)
