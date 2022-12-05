import torch
cuda = torch.cuda.is_available()
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
reg = 1e-2
reg_s = 1e-2

n = 1024
res = {}
n_trials = 10

for eps in eps_range:
    key = 'Epsilon = {}'.format(eps)
    res[key] = {'test_semi_dual': [], 'eval_l2_error_st': []}

# Define transport map
def transport_map(x):
    return x + 1/(6.0 - np.cos(np.pi*6*x)) - 0.2
def inv_map(x):
    y = np.random.rand(x.shape)
    for _ in range(50):
        deriv = 1 - (6*np.pi*np.sin(6*np.pi*y))/(6 - np.cos(6*np.pi*y))**2
        y -= (transport_map(y) - x)/deriv
    return y

for t in range(n_trials):
    print('Trial no {}'.format(t+1))

    # Generate source
    mu_train = np.random.rand(n*d).reshape(n, d)
    mu_test = np.random.rand(n*d).reshape(n, d)
    mu_eval = np.random.rand(n*d).reshape(n, d)

    # Generate target
    nu_train = transport_map(np.random.rand(n*d).reshape(n, d))
    nu_test = transport_map(np.random.rand(n*d).reshape(n, d))
    nu_eval = transport_map(np.random.rand(n*d).reshape(n, d))

    mu_train_torch = torch.tensor(mu_train, dtype=dtype).contiguous()
    nu_train_torch = torch.tensor(nu_train, dtype=dtype).contiguous()
    mu_test_torch = torch.tensor(mu_test, dtype=dtype).contiguous()
    nu_test_torch = torch.tensor(nu_test, dtype=dtype).contiguous()
    mu_eval_torch = torch.tensor(mu_eval, dtype=dtype).contiguous()
    if cuda:
        mu_train_torch = mu_train_torch.cuda()
        nu_train_torch = nu_train_torch.cuda()
        mu_test_torch = mu_test_torch.cuda()
        nu_test_torch = nu_test_torch.cuda()
        mu_eval_torch = mu_eval_torch.cuda()
    #nu = transport_map(torch.rand(int(1e8), d, dtype=dtype).cuda())
    #gt = (nu*inv_map(nu)).sum(1).mean()
    #print('Semi-dual ground truth: {}'.format(gt))

    for eps in eps_range:

        key = 'Epsilon = {}'.format(eps)
        print(key)
        print('\n')

        sinkhorn = SINKHORN(mu_train_torch, nu_train_torch, eps)
        sinkhorn.solve_dual(T=10000000, tol=1e-5)

        # Compute test semi primal
        f = lambda x: sinkhorn.f(x) + (reg/2)*(x**2).sum(-1)
        grad_f = lambda x: sinkhorn.grad_f(x) + reg*x
        if cuda:
            hess_f = lambda x: sinkhorn.hess_f(x) + reg_s*torch.eye(d).cuda()[None, :]
        else:
            hess_f = lambda x: sinkhorn.hess_f(x) + reg_s*torch.eye(d)[None, :]

        semi_dual_test = semi_dual(mu_test_torch, nu_test_torch, f, grad_f, 100.0, hess_f,
                                   T=1000000)
        print('Semi Dual test: {}'.format(semi_dual_test))
        res[key]['test_semi_dual'].append(semi_dual_test.item())

        # Compute L2 error source to target on eval set
        map_st_error = ((grad_f(mu_eval_torch).cpu().numpy().astype('double') - transport_map(mu_eval))**2).sum(1).mean()
        res[key]['eval_l2_error_st'].append(map_st_error.item())
        print('L2 error source/target: {}'.format(map_st_error))
        print('\n')

    print('###################################################################')
    print('\n')

    json_obj = json.dumps(res)
    f = open("Synth-XP/Tensorised/results/sinkhorn_n={}.json".format(n), "w")
    f.write(json_obj)
    f.close()
