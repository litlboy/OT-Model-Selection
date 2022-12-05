import numpy as np
import torch
cuda = torch.cuda.is_available()
from pykeops.torch import LazyTensor
from torch.autograd.functional import jacobian
from models import ICNNOT
from utils import fenchel_transform_newton
from scipy.optimize import minimize
import itertools
import json
from tqdm import tqdm
np.random.seed(0)
torch.manual_seed(0)

dtype = torch.float

d = 8
reg = 1e-3
list_n_neurons = [64, 128, 256]
list_lambd_cvx = [0.0, 0.001, 0.01, 0.1]
list_lambd_mean = [0.0, 0.001, 0.01, 0.1]

n = 1024
res = {}
n_trials = 10

anchors = 2*torch.rand(10, d) - 1.0
temp = .3
shifts = torch.randn(10)/temp
if cuda:
    anchors = anchors.cuda()
    shifts = sihts.cuda()
lazy_anchors = LazyTensor(anchors[None, :, :])
anchors_auto = (anchors[:, :, None]@anchors[:, None, :]).reshape(10, d**2)
ll_anchors = LazyTensor(anchors_auto[None, :, :])

def transport_map(x):
    pairs = LazyTensor(x[:, None, :])*lazy_anchors
    affine = pairs.sum(-1)/temp + LazyTensor(shifts[None, :, None])
    max_affine = affine.max(1)
    kern = (affine - max_affine[:, None]).exp()
    return (lazy_anchors*kern).sum(1)/kern.sum(1) + reg*x

def hess_map(x):
    pairs = LazyTensor(x[:, None, :])*lazy_anchors
    affine = pairs.sum(-1)/temp + LazyTensor(shifts[None, :, None])
    max_affine = affine.max(1)
    kern = (affine - max_affine[:, None]).exp()
    denom = kern.sum(1)
    dom_term = (ll_anchors*kern).sum(1)/denom
    nabla_f = (lazy_anchors*kern).sum(1)/denom
    corr_term = nabla_f[:, :, None]@nabla_f[:, None, :]
    hessian = (dom_term.reshape(x.shape[0], d, d) - corr_term)/temp
    return hessian + reg*torch.eye(d, dtype=dtype).cuda()[None, :, :]

def inv_map(x):
    return fenchel_transform_newton(x, transport_map, hess_map, T=20000, tol=1e-3,
                            self_conc=100.0)

for hyperparam in itertools.product(list_n_neurons, list_lambd_cvx, list_lambd_mean):
    key = '(N_NEURONS, LAMBDA_CVX, LAMBDA_MEAN) = {}'.format(hyperparam)
    res[key] = {'test_semi_dual': [], 'eval_l2_error_st': []}

for t in range(n_trials):
    print('Trial no {}'.format(t+1))

    # Generate source
    mu_train = torch.rand(n, d, dtype=dtype)
    mu_test = torch.rand(n, d, dtype=dtype)
    mu_eval = torch.rand(n, d, dtype=dtype)

    # Generate target
    mu_1 = torch.rand(n, d, dtype=dtype)
    mu_2 = torch.rand(n, d, dtype=dtype)
    mu_3 = torch.rand(n, d, dtype=dtype)

    if cuda:
        mu_train = mu_train.cuda()
        mu_test = mu_test.cuda()
        mu_eval = mu_eval.cuda()
        mu_1 = mu_1.cuda()
        mu_2 = mu_2.cuda()
        mu_3 = mu_3.cuda()

    nu_train = transport_map(mu_1)
    nu_test = transport_map(mu_2)
    nu_eval = transport_map(mu_3)

    for hyperparam in itertools.product(list_n_neurons, list_lambd_cvx, list_lambd_mean):

        print('(N_NEURONS, LAMBDA_CVX, LAMBDA_MEAN) = {}'.format(hyperparam))
        print('\n')

        icnnot = ICNNOT(n_neurons=hyperparam[0], lambd_cvx=hyperparam[1],
                        lambd_mean=hyperparam[2])
        icnnot.train(mu_train, nu_train)

        # Compute test semi primal
        f = lambda x: icnnot.convex_f(x)
        sum_f = lambda x: f(x).sum()
        grad = lambda x: jacobian(sum_f, x)

        # Compute L2 error source to target on eval set
        map_st_error = ((grad(mu_eval) + reg*mu_eval - transport_map(mu_eval))**2).sum(1).mean()
        key = '(N_NEURONS, LAMBDA_CVX, LAMBDA_MEAN) = {}'.format(hyperparam)
        res[key]['eval_l2_error_st'].append(map_st_error.item())
        print('L2 error source/target: {}'.format(map_st_error))

        if cuda:
            f_numpy = lambda x : f(torch.tensor(x[None, :],
                                dtype=dtype).cuda()).cpu().detach().numpy().astype('double')[0]
            grad_numpy = lambda x : grad(torch.tensor(x[None, :],
                                    dtype=dtype).cuda()).cpu().detach().numpy().astype('double')[0]
        else:
            f_numpy = lambda x : f(torch.tensor(x[None, :],
                                dtype=dtype)).detach().numpy().astype('double')[0]
            grad_numpy = lambda x : grad(torch.tensor(x[None, :],
                                    dtype=dtype)).detach().numpy().astype('double')[0]
        f_t = []
        for y in tqdm(nu_test, 'Computing Fenchel transform ...'):
            if cuda:
                g = lambda x: f_numpy(x) + reg*(x**2).sum()/2 - (y.cpu().detach().numpy().astype('double')*x).sum()
                grad_g = lambda x: grad_numpy(x) + reg*x - y.cpu().detach().numpy().astype('double')
            else:
                g = lambda x: f_numpy(x) + reg*(x**2).sum()/2 - (y.detach().numpy().astype('double')*x).sum()
                grad_g = lambda x: grad_numpy(x) + reg*x - y.detach().numpy().astype('double')

            prob = minimize(g, np.random.randn(d), jac=grad_g, tol=1e-3)
            while not prob['success']:
                prob = minimize(g, np.random.randn(d), jac=grad_g, tol=1e-3)
            f_t.append(prob['x'])

        f_t = torch.tensor(f_t, dtype=dtype)
        if cuda:
            f_t = f_t.cuda()
        mu_term = (f(mu_test) + reg*(mu_test**2).sum(1)).mean()
        nu_term = (f_t*nu_test).sum(1).mean() - (f(f_t) + reg*(f_t**2).sum(1)).mean()
        semi_dual_test = mu_term + nu_term
        print('Semi Dual test: {}'.format(semi_dual_test))
        res[key]['test_semi_dual'].append(semi_dual_test.item())

        print('\n')

    print('###################################################################')
    print('\n')

    json_obj = json.dumps(res)
    f = open("Synth-XP/Lse/results/icnnot.json", "w")
    f.write(json_obj)
    f.close()
