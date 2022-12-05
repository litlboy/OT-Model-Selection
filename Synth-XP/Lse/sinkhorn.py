import torch
cuda = torch.cuda.is_available()
from pykeops.torch import LazyTensor
from utils import fenchel_transform_newton
import numpy as np
from models import SINKHORN, semi_dual
import json
from tqdm import tqdm
from scipy.optimize import minimize
torch.manual_seed(0)
np.random.seed(0)

dtype = torch.float

d = 8
eps_range = [0.5, 0.1, 0.05, 0.01, 0.005]
reg = 1e-3

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

for eps in eps_range:
    key = 'Epsilon = {}'.format(eps)
    res[key] = {'test_semi_dual': [], 'eval_l2_error_st': []}

reg_s = 1e-2

for t in range(n_trials):
    print('Trial no {}'.format(t+1))
    print('\n')

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

    for eps in eps_range:

        key = 'Epsilon = {}'.format(eps)
        print(key)
        print('\n')

        sinkhorn = SINKHORN(mu_train, nu_train, eps)
        sinkhorn.solve_dual(T=10000000, tol=1e-5)

        # Compute test semi primal
        f = lambda x: sinkhorn.f(x) + (reg_s/2)*(x**2).sum(-1)
        grad_f = lambda x: sinkhorn.grad_f(x) + reg_s*x
        if cuda:
            hess_f = lambda x: sinkhorn.hess_f(x) + reg_s*torch.eye(d).cuda()[None, :]
        else:
            hess_f = lambda x: sinkhorn.hess_f(x) + reg_s*torch.eye(d)[None, :]

        semi_dual_test = semi_dual(mu_test, nu_test, f, grad_f, 100.0, hess_f,
                                   T=1000000)
        print('Semi Dual test: {}'.format(semi_dual_test))
        res[key]['test_semi_dual'].append(semi_dual_test.item())

        # Compute L2 error source to target on eval set
        map_st_error = ((grad_f(mu_eval) - transport_map(mu_eval))**2).sum(1).mean()
        res[key]['eval_l2_error_st'].append(map_st_error.item())
        print('L2 error source/target: {}'.format(map_st_error))
        print('\n')


    print('###################################################################')
    print('\n')

    json_obj = json.dumps(res)
    f = open("Synth-XP/Lse/results/sinkhorn_n={}.json".format(n), "w")
    f.write(json_obj)
    f.close()
