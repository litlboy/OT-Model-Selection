import numpy as np
import torch
cuda = torch.cuda.is_available()
from torch.autograd.functional import jacobian
from models import ICNNOT
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
#list_n_neurons = [256]
list_lambd_cvx = [0.0, 0.001, 0.01, 0.1]
list_lambd_mean = [0.0, 0.001, 0.01, 0.1]

n = 1024
res = {}
n_trials = 10

for hyperparam in itertools.product(list_n_neurons, list_lambd_cvx, list_lambd_mean):
    key = '(N_NEURONS, LAMBDA_CVX, LAMBDA_MEAN) = {}'.format(hyperparam)
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

    for hyperparam in itertools.product(list_n_neurons, list_lambd_cvx, list_lambd_mean):

        print('(N_NEURONS, LAMBDA_CVX, LAMBDA_MEAN) = {}'.format(hyperparam))
        print('\n')

        icnnot = ICNNOT(n_neurons=hyperparam[0], lambd_cvx=hyperparam[1],
                        lambd_mean=hyperparam[2])
        icnnot.train(mu_train_torch, nu_train_torch)

        # Compute test semi primal
        f = lambda x: icnnot.convex_f(x)
        sum_f = lambda x: f(x).sum()
        grad = lambda x: jacobian(sum_f, x)

        # Compute L2 error source to target on eval set
        if cuda:
            map_st_error = ((grad(mu_eval_torch).cpu().numpy().astype('double') + reg*mu_eval - transport_map(mu_eval))**2).sum(1).mean()
        else:
            map_st_error = ((grad(mu_eval_torch).numpy().astype('double') + reg*mu_eval - transport_map(mu_eval))**2).sum(1).mean()
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
        for y in tqdm(nu_test_torch, 'Computing Fenchel transform ...'):
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
        mu_term = (f(mu_test_torch) + reg*(mu_test_torch**2).sum(1)).mean()
        nu_term = (f_t*nu_test_torch).sum(1).mean() - (f(f_t) + reg*(f_t**2).sum(1)).mean()
        semi_dual_test = mu_term + nu_term
        print('Semi Dual test: {}'.format(semi_dual_test))
        res[key]['test_semi_dual'].append(semi_dual_test.item())

        print('\n')

    print('###################################################################')
    print('\n')

    json_obj = json.dumps(res)
    f = open("Synth-XP/Tensorised/results/icnnot.json", "w")
    f.write(json_obj)
    f.close()
