import logging

import cvxpy as cp
import numpy as np
import torch
from torch.utils.data import DataLoader

import utils.data_utils as datu
import utils.misc_utils as mscu

logger = logging.getLogger("online_label_shift")


def get_dirichlet_marginal(alpha, seed):
    np.random.seed(seed)
    return np.random.dirichlet(alpha)


def get_label_marginals(labels, num_labels=None):
    seen_labels, seen_counts = np.unique(labels, return_counts=True)
    seen_labels = seen_labels.astype(int)

    num_labels = np.max(labels) + 1 if num_labels is None else num_labels
    all_counts = np.zeros(num_labels)
    for idx, label in enumerate(seen_labels):
        all_counts[label] = seen_counts[idx]

    return all_counts / np.sum(all_counts)


def get_accuracy(y_true, y_pred):
    y_true, y_pred = mscu.np2torch(y_true), mscu.np2torch(y_pred)
    return torch.mean((y_true == y_pred).float(), dtype=torch.float)


def sample_idc_with_replacement(labels, num_samples, target_marginals, num_labels=None):
    num_labels = np.max(labels) + 1 if num_labels is None else num_labels
    indices_by_label = [(labels == y).nonzero()[0] for y in range(num_labels)]

    label_choices = np.argmax(
        np.random.multinomial(1, target_marginals, num_samples), axis=1
    )

    # sample an example from X with replacement
    target_idc = []
    for i in range(num_samples):
        idx = np.random.choice(indices_by_label[label_choices[i]])
        target_idc.append(idx)

    return target_idc


def sample_idc_wo_replacement(labels, num_samples, target_marginals, num_labels=None):
    labels = mscu.torch2np(labels)
    num_labels = np.max(labels) + 1 if num_labels is None else num_labels
    indices_by_label = [(labels == y).nonzero()[0] for y in range(num_labels)]

    label_choices = np.argmax(
        np.random.multinomial(1, target_marginals, num_samples), axis=1
    )
    labels, label_counts = np.unique(label_choices, return_counts=True)

    # sample examples from X without replacement
    target_idc = []
    for label_idx, label in enumerate(labels):
        idc = np.random.choice(
            indices_by_label[label], size=label_counts[label_idx], replace=False
        )
        target_idc.extend(idc)
    np.random.shuffle(target_idc)

    return target_idc


def get_reweighted_predictions(probs, source_marginals, target_marginals):
    weight = target_marginals / source_marginals
    new_probs = weight[None] * probs
    preds = np.argmax(new_probs, axis=-1)
    return preds


def idx2onehot(y, num_labels):
    y = y.astype(int)
    b = np.zeros((y.size, num_labels))
    b[np.arange(y.size), y] = 1
    return b


def get_soft_confusion_matrix(y_true, y_prob, num_labels):
    # Input y_prob is output probabilities in forms of n by k num_labels
    y_true, y_prob = mscu.torch2np(y_true), mscu.torch2np(y_prob)

    n, _ = np.shape(y_prob)
    C = np.dot(y_prob.T, idx2onehot(y_true, num_labels))
    return C / n


def get_confusion_matrix(y_true, y_pred, num_labels):
    y_true, y_pred = mscu.torch2np(y_true), mscu.torch2np(y_pred)
    n_samples = y_true.size

    C = np.dot(idx2onehot(y_pred, num_labels).T, idx2onehot(y_true, num_labels))
    return C / n_samples


def BBSE(
    confusion_matrix,
    source_train_marginals,
    y_pred_marginals=None,
    y_pred=None,
    lamb=1e-8,
):
    if y_pred_marginals is None:
        num_labels = confusion_matrix.shape[0]
        y_pred_marginals = get_label_marginals(y_pred, num_labels)

    num_labels = confusion_matrix.shape[0]
    wt = np.linalg.solve(
        np.dot(confusion_matrix.T, confusion_matrix) + lamb * np.eye(num_labels),
        np.dot(confusion_matrix.T, y_pred_marginals),
    )
    wt = np.squeeze(wt)
    wt[wt < 0.0] = 0.0

    marginal_est = np.multiply(wt, source_train_marginals)
    return marginal_est / np.sum(marginal_est)


def MLLS(
    source_pred_soft_marginals, test_pred_prob, numClasses, tol=1e-6, max_iter=10000
):
    def EM(p_base, soft_probs, nclass):
        #   Initialization
        q_prior = np.ones(nclass)
        q_posterior = np.copy(soft_probs)
        curr_q_prior = np.average(soft_probs, axis=0)
        iter = 0
        while abs(np.sum(abs(q_prior - curr_q_prior))) >= tol and iter < max_iter:
            q_prior = np.copy(curr_q_prior)
            temp = np.multiply(np.divide(curr_q_prior, p_base), soft_probs)
            q_posterior = np.divide(temp, np.expand_dims(np.sum(temp, 1), axis=1))
            curr_q_prior = np.average(q_posterior, axis=0)
            iter += 1
        return curr_q_prior

    py_target = EM(source_pred_soft_marginals, test_pred_prob, numClasses)

    return py_target


def RLLS(
    confusion_matrix, mu_y, mu_train_y, num_labels, n_train, rho_coeff=0.01, delta=0.05
):
    """Estimate current label marginals"""

    def compute_3deltaC(n_class, n_train, delta):
        rho = 3 * (
            2 * np.log(2 * n_class / delta) / (3 * n_train)
            + np.sqrt(2 * np.log(2 * n_class / delta) / n_train)
        )
        return rho

    def compute_w_opt(C_yy, mu_y, mu_train_y, rho):
        n = C_yy.shape[0]
        theta = cp.Variable(n)
        b = mu_y - mu_train_y
        objective = cp.Minimize(cp.pnorm(C_yy @ theta - b) + rho * cp.pnorm(theta))
        constraints = [-1 <= theta]
        prob = cp.Problem(objective, constraints)

        # The optimal objective value is returned by `prob.solve()`.
        result = prob.solve()
        # The optimal value for x is stored in `x.value`.
        w = 1 + theta.value
        return w

    rho = rho_coeff * compute_3deltaC(num_labels, n_train, delta=delta)
    wt_hard = compute_w_opt(confusion_matrix, mu_y, mu_train_y, rho)

    marginal_est = np.multiply(wt_hard, mu_train_y)
    marginal_est[marginal_est < 0.0] = 0.0
    return marginal_est / np.sum(marginal_est)


def get_random_probability(n):
    """Return uniformly random vector simplex
    Taken from https://stackoverflow.com/questions/65154622/sample-uniformly-at-random-from-a-simplex-in-python
    """
    k = np.random.exponential(scale=1.0, size=n)
    return k / sum(k)


### Functions for generating shifts


def generate_tweak_one_shift(num_labels, probabilities, idx_to_tweak=0):
    shifted_label_marginals = []
    for prob in probabilities:
        assert prob >= 0.0 and prob <= 1.0
        other_prob = (1 - prob) / (num_labels - 1)
        cur_label_marginals = np.ones(num_labels) * other_prob
        cur_label_marginals[idx_to_tweak] = prob
        shifted_label_marginals.append(cur_label_marginals)
    return shifted_label_marginals


def generate_monotone_shift(marg1, marg2, total_time):
    return [
        (1 - t / total_time) * marg1 + (t / total_time) * marg2 for t in range(total_time)
    ]


def generate_constant_shift(marg_target, total_time):
    return [np.copy(marg_target) for _ in range(total_time)]


def generate_square_shift(marg1, marg2, total_time, period=None):
    if period is None:
        period = int(total_time ** (1 / 2))
    shift_generator = (
        lambda t: np.copy(marg1) if (t // period) % 2 == 0 else np.copy(marg2)
    )
    return [shift_generator(t) for t in range(total_time)]


def generate_sinusoidal_shift(marg1, marg2, total_time, period=None):
    if period is None:
        period = int(total_time ** (1 / 2))
    coeff_generator = lambda t: np.sin((t % period) * np.pi / period)
    return [
        (1 - coeff_generator(t)) * marg1 + coeff_generator(t) * marg2
        for t in range(total_time)
    ]


def generate_bernouli_shift(marg1, marg2, total_time, change_prob=None):
    if change_prob is None:
        change_prob = 1 / (total_time ** (1 / 2))

    a_t = 0
    shifts = []
    for t in range(total_time):
        cur_marg = (1 - a_t) * marg1 + a_t * marg2
        shifts.append(cur_marg)

        if np.random.choice((0, 1), p=(1 - change_prob, change_prob)) == 1:
            a_t = 1 - a_t
    return shifts


SHIFT_FUNCTIONS = {
    "monotone": generate_monotone_shift,
    "tweak_one": generate_tweak_one_shift,
    "constant": generate_constant_shift,
    "square": generate_square_shift,
    "sinusoidal": generate_sinusoidal_shift,
    "bernouli": generate_bernouli_shift,
}


def get_shifts(shift_type, **params):
    if shift_type not in SHIFT_FUNCTIONS.keys():
        raise ValueError(f"Shift {shift_type} not supported")
    shifts = SHIFT_FUNCTIONS[shift_type](**params)
    return shifts


def get_shifted_dataloaders(
    dataset, marginal_shifts, num_samples, batch_size=32, seed=None, num_labels=None
):
    if seed is not None:
        mscu.set_seed(seed)

    target_dataloaders = []
    for target_marginals in marginal_shifts:
        target_idc = sample_idc_wo_replacement(
            dataset.y_array, num_samples, target_marginals, num_labels=num_labels
        )
        target_dataset = datu.LabelShiftedSubset(
            dataset, indices=target_idc, sampling_target_marginals=target_marginals
        )
        target_dataloader = DataLoader(target_dataset, batch_size=batch_size)
        target_dataloaders.append(target_dataloader)
    return target_dataloaders


def get_l2_square_diff(estimated_margs, true_margs):
    diff = estimated_margs - true_margs
    l2_sq = np.sum(diff ** 2)
    return l2_sq
