import numpy as np
import torch
from utils import fenchel_transform_newton
from pykeops.torch import LazyTensor
from models import SSNB
from scipy.stats import ortho_group
from utils import semi_dual
import json
np.random.seed(0)
torch.manual_seed(0)

d = 8
l_range = [0.2, 0.5, 0.7, 0.9]
L_range = [0.3, 0.5, 0.7, 0.9, 1.3]
reg = 1e-3

res = {}

hyperparams = []
for l in l_range:
    for L in L_range:
        if l<L:
            hyperparams.append((l, L))
            key = '(l, L) = {}'.format((l, L))
            res[key] = {'test_semi_dual': [], 'eval_l2_error_st': []}

#n = 1024
n = 10
n_trials = 10

anchors = 2*torch.rand(10, d) - 1.0
lazy_anchors = LazyTensor(anchors[None, :, :])
anchors_auto = (anchors[:, :, None]@anchors[:, None, :]).reshape(10, d**2)
ll_anchors = LazyTensor(anchors_auto[None, :, :])
temp = .3
shifts = torch.randn(10)/temp

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
    return hessian + reg*torch.eye(d, dtype=dtype)[None, :, :]

def inv_map(x):
    return fenchel_transform_newton(x, transport_map, hess_map, T=20000, tol=1e-3,
                            self_conc=100.0)

for t in range(n_trials):
    print('Trial no {}'.format(t+1))

    # Generate source
    mu_train = torch.rand(n, d)
    mu_test = torch.rand(n, d)
    mu_eval = torch.rand(n, d)

    # Generate target
    nu_train = transport_map(torch.rand(n, d))
    nu_test = transport_map(torch.rand(n, d))
    nu_eval = transport_map(torch.rand(n, d))

    for (l, L) in hyperparams:

        key = '(l, L) = {}'.format((l, L))
        print(key)
        print('\n')

        ssnb = SSNB(mu_train.numpy().astype('double'), l, L)
        ssnb.train(nu_train.numpy().astype('double'), 10)

        # Compute test semi primal
        f = lambda x: ssnb.f_grad_f(x)[0]
        grad = lambda x: ssnb.f_grad_f(x)[1]
        semi_dual_test = semi_dual(mu_test.numpy().astype('double'),
                                    nu_test.numpy().astype('double'), f,
                                    grad, 1/(2*L), T=200)
        print('Semi Dual test: {}'.format(semi_dual_test))
        res[key]['test_semi_dual'].append(semi_dual_test.item())

        # Compute L2 error source to target on eval set
        map_st_error = ((grad(mu_eval.numpy().astype('double')) - transport_map(mu_eval).numpy().astype('double'))**2).sum(1).mean()
        res[key]['eval_l2_error_st'].append(map_st_error.item())
        print('L2 error source/target: {}'.format(map_st_error))
        print('\n')

    print('###################################################################')
    print('\n')

    json_obj = json.dumps(res)
    f = open("Synth-XP/Lse/results/ssnb.json", "w")
    f.write(json_obj)
    f.close()
