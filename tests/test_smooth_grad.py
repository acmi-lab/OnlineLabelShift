"""This script tests the implemention of smoothed gradient"""
import init_path
import numpy as np

import utils.misc_utils as mscu


def l2_norm(x):
    return x.T @ x


n_params = 5
delta = 1e-3
k = 4

x0 = np.random.randn(n_params)
smooth_grad = mscu.finite_gradient(l2_norm, x0, delta, k)
grad = 2 * x0

print(f"{smooth_grad=}")
print(f"{grad=}")
