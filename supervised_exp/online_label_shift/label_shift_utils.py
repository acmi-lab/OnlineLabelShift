import math
import sys
from ctypes import c_short

import cvxpy as cp
import numpy as np
from cvxopt import matrix, solvers
import copy 

solvers.options['show_progress'] = False
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger("label_shift")


def np2torch(array):
    if type(array) != torch.Tensor:
        return torch.from_numpy(array)
    return array


def get_dirichlet_marginal(alpha, seed): 
    
    np.random.seed(seed)
    
    idx = np.where(alpha > 0)[0]
    
    target_y_dist = np.zeros_like(alpha)
    target_y_dist[idx] = np.random.dirichlet(alpha[idx])

    return target_y_dist

def get_resampled_indices(y, num_labels, Py, seed):

    np.random.seed(seed)
    # get indices for each label
    indices_by_label = [(y==k).nonzero()[0] for k in range(num_labels)]
    num_samples = int(min([len(indices_by_label[i])/Py[i] for i in range(num_labels)]))

    agg_idx = []        
    for i in range(num_labels):
        # sample an example from X with replacement
        idx = np.random.choice(indices_by_label[i], size = int(num_samples* Py[i]), replace = False)
        agg_idx.append(idx)

    return np.concatenate(agg_idx)

def tweak_dist_idx(y, num_labels, n, Py, seed):

    np.random.seed(seed)
    # get indices for each label
    indices_by_label = [(y==k).nonzero()[0] for k in range(num_labels)]
    
    labels = np.argmax(
        np.random.multinomial(1, Py, n), axis=1)

    agg_idx = []        
    for i in range(n):
        # sample an example from X with replacement
        idx = np.random.choice(indices_by_label[labels[i]])
        agg_idx.append(idx)

    return agg_idx

def compute_w_opt(C_yy,mu_y,mu_train_y, rho):
    n = C_yy.shape[0]
    theta = cp.Variable(n)
    b = mu_y - mu_train_y
    objective = cp.Minimize(cp.pnorm(C_yy @ theta - b) + rho* cp.pnorm(theta))
    constraints = [-1 <= theta]
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    # The optimal value for x is stored in `x.value`.
    # print(theta.value)
    w = 1 + theta.value
#     print('Estimated w is', w)
    #print(constraints[0].dual_value)
    return w

def im_weights_update(source_y, target_y, cov, im_weights=None, ma = 0.5):
    """
    Solve a Quadratic Program to compute the optimal importance weight under the generalized label shift assumption.
    :param source_y:    The marginal label distribution of the source domain.
    :param target_y:    The marginal pseudo-label distribution of the target domain from the current classifier.
    :param cov:         The covariance matrix of predicted-label and true label of the source domain.
    :return:
    """
    # Convert all the vectors to column vectors.
    dim = cov.shape[0]
    source_y = source_y.reshape(-1, 1).astype(np.double)
    target_y = target_y.reshape(-1, 1).astype(np.double)
    cov = cov.astype(np.double)

    P = matrix(np.dot(cov.T, cov), tc="d")
    q = -matrix(np.dot(cov.T, target_y), tc="d")
    G = matrix(-np.eye(dim), tc="d")
    h = matrix(np.zeros(dim), tc="d")
    A = matrix(source_y.reshape(1, -1), tc="d")
    b = matrix([1.0], tc="d")
    sol = solvers.qp(P, q, G, h, A, b)
    new_im_weights = np.array(sol["x"])
    
    # import pdb; pdb.set_trace()

    # EMA for the weights
    im_weights = (1 - ma) * new_im_weights + ma * im_weights

    return im_weights

def compute_3deltaC(n_class, n_train, delta):
    rho = 3*(2*np.log(2*n_class/delta)/(3*n_train) + np.sqrt(2*np.log(2*n_class/delta)/n_train))
    return rho


def EM(p_base,soft_probs, nclass): 
#   Initialization
    q_prior = np.ones(nclass)
    q_posterior = np.copy(soft_probs)
#     print (Q_func(q_posterior,q_prior))
    curr_q_prior = np.average(soft_probs,axis=0)
#     print q_prior
#     print curr_q_prior
    iter = 0
    while abs(np.sum(abs(q_prior - curr_q_prior))) >= 1e-6 and iter < 10000:
#         print iter
        q_prior = np.copy( curr_q_prior)
#         print curr_q_prior
#         print np.divide(curr_q_prior, p_base)
        temp = np.multiply(np.divide(curr_q_prior, p_base), soft_probs)
#         print temp
        q_posterior = np.divide(temp, np.expand_dims(np.sum(temp,1),axis=1))
#         print q_posterior
        curr_q_prior = np.average(q_posterior, axis = 0)
#         print curr_q_prior
        iter +=1 
#     print q_prior
#     print curr_q_prior
#     print (Q_func(q_posterior,curr_q_prior))
#     print iter
    return curr_q_prior 


def get_fisher(py_x, py, w):
    
    dims = py.shape[0] -1 
    temp = np.divide(py_x,py)
#     print temp[:,-1].shape
#     print (temp[:,:dims] - temp[:,-1]).shape
    score = np.divide(temp[:,:dims] - np.expand_dims(temp[:,-1], axis=1),np.expand_dims(np.matmul(py_x,w),axis=1))
#     print score.shape
    fisher = np.matmul(score.T,score)
    
    return fisher

def idx2onehot(a,k):
    a=a.astype(int)
    b = np.zeros((a.size, k))
    b[np.arange(a.size), a] = 1
    return b

def confusion_matrix(ytrue, ypred,k):
    # C[i,j] denotes the frequency of ypred = i, ytrue = j.
    n = ytrue.size
    C = np.dot(idx2onehot(ypred,k).T,idx2onehot(ytrue,k))
    return C/n

def confusion_matrix_probabilistic(ytrue, ypred,k):
    # Input is probabilistic classifiers in forms of n by k matrices
    n,d = np.shape(ypred)
    C = np.dot(ypred.T, idx2onehot(ytrue,k))
    return C/n

def calculate_marginal(y,k):
    mu = np.zeros(shape=(k))
    for i in range(k):
        mu[i] = np.count_nonzero(y == i)
    return mu/len(y)

def calculate_marginal_probabilistic(y,k):
    return np.mean(y,axis=0)

def estimate_labelshift_ratio(ytrue_s, ypred_s, ypred_t,k):
    if ypred_s.ndim == 2: # this indicates that it is probabilistic
        C = confusion_matrix_probabilistic(ytrue_s,ypred_s,k)
        mu_t = calculate_marginal_probabilistic(ypred_t, k)
    else:
        C = confusion_matrix(ytrue_s, ypred_s,k)
        mu_t = calculate_marginal(ypred_t, k)
    lamb = 1e-8
    wt = np.linalg.solve(np.dot(C.T, C)+lamb*np.eye(k), np.dot(C.T, mu_t))
    return wt

def estimate_labelshift_ratio_direct(ytrue_s, ypred_s, ypred_t,k):
    if ypred_s.ndim == 2: # this indicates that it is probabilistic
        C = confusion_matrix_probabilistic(ytrue_s,ypred_s,k)
        mu_t = calculate_marginal_probabilistic(ypred_t, k)
    else:
        C = confusion_matrix(ytrue_s, ypred_s,k)
        mu_t = calculate_marginal(ypred_t, k)
    # lamb = (1/min(len(ypred_s),len(ypred_t)))
    wt = np.linalg.solve(C,mu_t)
    return wt

def estimate_target_dist(wt, ytrue_s,k):
    ''' Input:
    - wt:    This is the output of estimate_labelshift_ratio)
    - ytrue_s:      This is the list of true labels from validation set

    Output:
    - An estimation of the true marginal distribution of the target set.
    '''
    mu_t = calculate_marginal(ytrue_s,k)
    return wt*mu_t

# functions that convert beta to w and converge w to a corresponding weight function.
def beta_to_w(beta, y, k):
    w = []
    for i in range(k):
        w.append(np.mean(beta[y.astype(int) == i]))
    w = np.array(w)
    return w

# a function that converts w to beta.
def w_to_beta(w,y):
    return w[y.astype(int)]

def w_to_weightfunc(w):
    return lambda x, y: w[y.astype(int)]


def get_label_marginals(labels, num_labels=None):
    seen_labels, seen_counts = np.unique(labels, return_counts=True)
    seen_labels = seen_labels.astype(int)

    num_labels = np.max(labels) + 1 if num_labels is None else num_labels
    all_counts = np.zeros(num_labels)
    for idx, label in enumerate(seen_labels):
        all_counts[label] = seen_counts[idx]

    return all_counts / np.sum(all_counts)

def get_marginal_estimate(confusion_matrix, mu_train_y, y_pred_marginals=None, y_pred=None):
    if y_pred_marginals is None:
        num_labels = confusion_matrix.shape[0]
        y_pred_marginals = get_label_marginals(y_pred, num_labels)
    
    num_labels = confusion_matrix.shape[0]
    lamb = 1e-8
    wt = np.linalg.solve(np.dot(confusion_matrix.T, confusion_matrix)+lamb*np.eye(num_labels), np.dot(confusion_matrix.T, y_pred_marginals))
    wt = np.squeeze(wt)
        
    wt[wt < 0.0] = 0.0    
    
    marginal_est = np.multiply(wt, mu_train_y)
    
    return marginal_est/np.sum(marginal_est)
    

def MLLS(ypred_source, ytrue_source, ypred_target, numClasses): 

    # ypred_hard_source = np.argmax(ypred_source, 1)

    # ypred_marginal =  calculate_marginal(ypred_hard_source,numClasses)
    ypred_marginal = np.average(ypred_source, axis=0)
    # logger.debug(f"{ypred_marginal}")
    # logger.debug(f"{ypred_target}") 

    py_target = EM(ypred_marginal, ypred_target, numClasses)

    return py_target


def RLLS(
    confusion_matrix, mu_y, mu_train_y, num_labels, n_train, rho_coeff=0.01, delta=0.05
):
    """Estimate current label marginals
    """

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
    wt = compute_w_opt(confusion_matrix, mu_y, mu_train_y, rho)
    wt = np.squeeze(wt)
    
    marginal_est = np.multiply(wt, mu_train_y)
    marginal_est[marginal_est < 0.0] = 0.0
    return marginal_est / np.sum(marginal_est)



def gan_loss(epoch, disc_net, gen_params, opt_d, opt_g, valloader, val_labels, testloader, val_target_dist, device):
    disc_net.train()
#     gen_params.train()

    val_inputs = torch.tensor(valloader).float().to(device)
    test_inputs = torch.tensor(testloader).float().to(device)

    real_labels = torch.ones(valloader.shape[0]).long().to(device)
    fake_labels = torch.zeros(testloader.shape[0]).long().to(device)

    
    
    im_weights =  torch.divide(F.softmax(gen_params, dim = -1), torch.tensor(val_target_dist).float().to(device))
#     print(im_weights)
#     print(val_inputs[:10])
    
#     val_inputs = torch.mul(val_inputs, im_weights) 
#     print(torch.sum(test_inputs,dim=-1).shape)
#     val_inputs = val_inputs/ torch.sum(val_inputs,dim=-1, keepdim=True)
#     print(val_inputs[:10])
    
    for i in range(1):
        opt_d.zero_grad()

        inputs = torch.cat((val_inputs.detach(), test_inputs), 0)
        labels = torch.cat((real_labels, fake_labels), 0)
        weights = torch.cat([im_weights[val_labels].detach(), torch.ones_like(fake_labels)],0)
        xx = disc_net(inputs)
        d_loss = torch.mean(nn.CrossEntropyLoss(reduction="none")(xx, labels)*weights)
        
#         if epoch %50 ==0 : 
#             print(xx)
        d_loss.backward()
        opt_d.step()
    
    opt_g.zero_grad()
    
#     print(type(real_labels))
#     print(type(test_inputs))
#     print(test_inputs.shape)
    
    g_loss = -1.0*torch.mean(nn.CrossEntropyLoss(reduction="none")(disc_net(val_inputs), real_labels)*im_weights[val_labels])

    g_loss.backward()
    opt_g.step()

    return g_loss.item(), d_loss.item()


def gan_target_marginal(epochs, ypred_source, ytrue_val, ypred_target, numClasses, device="cpu"): 

    py_true_source = calculate_marginal(ytrue_val, numClasses).reshape((numClasses))
    d_net = nn.Sequential(nn.Linear(numClasses, 2, bias=False) ) 
    d_net = d_net.to(device)
    opt_d = optim.SGD(d_net.parameters(), lr=.1, momentum= 0.0, weight_decay=0.000)

    g_params = torch.Tensor([ 0.0]* numClasses).requires_grad_()
    opt_g = optim.SGD([g_params], lr=.10, momentum= 0.0, weight_decay=0.000)

    for epoch in range(epochs):
        
        g_loss, d_loss = gan_loss(epoch, d_net, g_params, opt_d, opt_g, ypred_source, ytrue_val, ypred_target, py_true_source, device)
        
        # if epoch % 50 == 0: 
            # print(f"Epoch: {epoch:.2f} G Loss: {g_loss:.5f}, D Loss: {d_loss:.5f}")
            # print (F.softmax(g_params, dim = -1))
            
    return F.softmax(g_params, dim = -1).detach().numpy()


#----------------------------------------------------------------------------

def estimation_err(estimated_marginal, true_marginal): 
    return np.linalg.norm(estimated_marginal - true_marginal, ord=2)**2

def im_reweight_acc(im_weights, probs, targets, map_preds = None): 

    new_probs = im_weights[None]*probs

    preds = np.argmax(new_probs, axis=-1)
    
    if map_preds is not None: 
        return np.mean(map_preds(preds) == map_preds(targets))*100
    else: 
        return  np.mean(preds == targets)*100


def get_acc(probs, labels, map_preds = None): 
    preds = np.argmax(probs, axis=-1)
    
    if map_preds is not None:
        return np.mean(map_preds(preds) == map_preds(labels))*100
    else: 
        return np.mean(preds == labels)*100
    


def generate_tweak_one_shift(num_labels, probabilities, idx_to_tweak=0):
    shifted_label_marginals = []
    for prob in probabilities:
        assert prob >= 0.0 and prob <= 1.0
        other_prob = (1 - prob) / (num_labels - 1)
        cur_label_marginals = np.ones(num_labels) * other_prob
        cur_label_marginals[idx_to_tweak] = prob
        shifted_label_marginals.append(cur_label_marginals)
    return shifted_label_marginals


def generate_monotone_shift(marg1, marg2, total_time, period=None):
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

def load_default(num_samples, num_classes, seed=42, alpha=10.0):
   
    np.random.seed(seed)
    
    if num_samples == 1:
        marg = np.random.dirichlet(alpha * np.ones(num_classes))
        return marg
        
        
    elif num_samples == 2:
        # marg1 = np.random.dirichlet(alpha * np.ones(num_classes))
        # marg2 = np.random.dirichlet(alpha * np.ones(num_classes))
        marg1 = np.array([1.0/num_classes]*num_classes)
        marg2 = np.zeros(num_classes)
        marg2[0] = 1.0
        return marg1, marg2


def generate_shifts(label_shift_func, label_shift_kwargs, total_time, num_classes, seed=42, alpha=10.0): 
    
    if "marg1" in label_shift_kwargs:
        if label_shift_kwargs["marg1"] is None: 
            label_shift_kwargs["marg1"], label_shift_kwargs["marg2"] = load_default(2, num_classes, seed=seed, alpha=alpha)
    
        logger.info(f"marg1 = {label_shift_kwargs['marg1']} \n marg2 = {label_shift_kwargs['marg2']}")

    elif "marg_target" in label_shift_kwargs:
        if label_shift_kwargs["marg_target"] is None: 
            label_shift_kwargs["marg_target"] = load_default(1, num_classes, seed=seed, alpha=alpha)
    
    
    return label_shift_func(total_time=total_time, **label_shift_kwargs)


class ShiftEstimatorWithMemory:
    def __init__(
        self,
        source_marginal,
        marginal_estimator_name,
    ) -> None:
        self.marginal_est = source_marginal
        self.marginal_estimator_name = marginal_estimator_name
        
    def get_marginal_estimate(self, marginal):
        cur_marginal_est = copy.deepcopy(self.marginal_est)
        self.update_marginal_estimate(marginal)
        return cur_marginal_est

    def update_marginal_estimate(self, marginal):
        raise NotImplementedError


class FollowTheHistoryEstimator(ShiftEstimatorWithMemory):
    def __init__(
        self,
        source_marginal,
        marginal_estimator_name="FTH",
    ) -> None:
        super().__init__(
            source_marginal,
            marginal_estimator_name=marginal_estimator_name,
        )
        self.marginal_est_history = None

    def update_marginal_estimate(self, marginal):
        cur_marginal_est = marginal
        if self.marginal_est_history is None:
            self.marginal_est_history = np2torch(cur_marginal_est).unsqueeze(0)
        else:
            self.marginal_est_history = torch.vstack(
                (self.marginal_est_history, np2torch(cur_marginal_est))
            )

        self.marginal_est = torch.mean(self.marginal_est_history, dim=0).numpy()


class FollowTheFixedWindowHistoryEstimator(FollowTheHistoryEstimator):
    def __init__(
        self, source_marginal, window_size
    ) -> None:
        super().__init__(
            source_marginal,
            marginal_estimator_name="FTFWH",
        )
        self.num_records = 0
        self.window_size = window_size

    def update_marginal_estimate(self, marginals):
        if self.num_records >= int(self.window_size):
            self.marginal_est_history = self.marginal_est_history[1:, :]
        super().update_marginal_estimate(marginals)
        self.num_records += 1


class FollowLeadingHistoryFollowTheLeaderEstimator(ShiftEstimatorWithMemory):
    def __init__(
        self,
        source_marginal,
        num_labels,
        meta_lr=None,
    ) -> None:
        super().__init__(
            source_marginal, marginal_estimator_name="FLH-FTL"
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

    def update_marginal_estimate(self, marginal):
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
        cur_marginal_est = marginal
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



### Unsupervised estimators

class MarginalEstimator:
    def __init__(self, marginal_estimator_name=None) -> None:
        self.marginal_estimator_name = marginal_estimator_name

    def get_marginal_estimate(self, pred_marginals, confusion_matrix):
        raise NotImplementedError


class LocalShiftEstimator(MarginalEstimator):
    def __init__(
        self,
        marginal_estimator_name=None,
    ) -> None:
        super().__init__(marginal_estimator_name=marginal_estimator_name)

    def get_marginal_estimate(self, pred_marginals, confusion_matrix):
        return self._get_marginal_estimate(pred_marginals=pred_marginals, confusion_matrix=confusion_matrix)

    def _get_marginal_estimate(self, pred_marginals, confusion_matrix):
        raise NotImplementedError


class BlackBoxShiftEstimator(LocalShiftEstimator):
    def __init__(self, source_marginal) -> None:
        super().__init__(
            marginal_estimator_name="BBSE",
        )

        self.source_marginal = source_marginal

    def _get_marginal_estimate(self, pred_marginals, confusion_matrix):
        marginal_est = get_marginal_estimate(
            confusion_matrix=confusion_matrix, mu_train_y=self.source_marginal , y_pred_marginals=pred_marginals
        )
        return marginal_est


class RegularizedShiftEstimator(LocalShiftEstimator):
    def __init__(self, source_marginal, num_classes, n_train) -> None:
        super().__init__(
            marginal_estimator_name="RLLS",
        )
        self.source_marginal = source_marginal
        self.num_classes = num_classes
        self.n_train = n_train

    def _get_marginal_estimate(self, pred_marginals, confusion_matrix):
        marginal_est = RLLS(
            confusion_matrix=confusion_matrix,
            mu_y=pred_marginals,
            mu_train_y=self.source_marginal,
            num_labels=self.num_classes,
            n_train=self.n_train,
        )
        return marginal_est
