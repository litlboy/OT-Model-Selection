import torch
import numpy as np
from models import SINKHORN, semi_dual
import json
from tqdm import tqdm
from scipy.optimize import minimize
from timeit import timeit
torch.manual_seed(0)
np.random.seed(0)

dtype = torch.float

d = 8
reg = 0.1
n = 1000

res = {'1st order scheme': [], '2nd order scheme': []}
list_eps = [0.1, 0.05, 0.01, 0.005]

# Define transport map
def transport_map(x):
    return x + 1/(6.0 - np.cos(np.pi*6*x)) - 0.2
def inv_map(x):
    y = np.random.rand(x.shape)
    for _ in range(50):
        deriv = 1 - (6*np.pi*np.sin(6*np.pi*y))/(6 - np.cos(6*np.pi*y))**2
        y -= (transport_map(y) - x)/deriv
    return y

for eps in list_eps:

    print('TEMPERATURE: {}'.format(eps))
    print('\n')

    # Generate source
    mu_train = np.random.rand(n*d).reshape(n, d)
    mu_test = np.random.rand(n*d).reshape(n, d)
    mu_eval = np.random.rand(n*d).reshape(n, d)

    # Generate target
    nu_train = transport_map(np.random.rand(n*d).reshape(n, d))
    nu_test = transport_map(np.random.rand(n*d).reshape(n, d))
    nu_eval = transport_map(np.random.rand(n*d).reshape(n, d))


    sinkhorn = SINKHORN(torch.tensor(mu_train, dtype=dtype).cuda().contiguous(),
                        torch.tensor(nu_train, dtype=dtype).cuda().contiguous(), eps)
    sinkhorn.solve_dual(T=10000000, tol=1e-5)

    # Compute test semi primal
    f = lambda x: sinkhorn.f(x) + (reg/2)*(x**2).sum(-1)
    grad_f = lambda x: sinkhorn.grad_f(x) + reg*x
    hess_f = lambda x: sinkhorn.hess_f(x) + reg*torch.eye(d).cuda()[None, :]
    max_norm = ((nu_train**2).sum(1)).max()/eps

    first_order = lambda: semi_dual(torch.tensor(mu_test, dtype=dtype).cuda().contiguous(),
                               torch.tensor(nu_test, dtype=dtype).cuda().contiguous(),
                               f, grad_f, 1/(2*(max_norm + reg)), T=1000000, tol=1e-3)
    second_order = lambda: semi_dual(torch.tensor(mu_test, dtype=dtype).cuda().contiguous(),
                               torch.tensor(nu_test, dtype=dtype).cuda().contiguous(),
                               f, grad_f, d**0.5/eps, hess_f, T=1000000, tol=1e-3)

    time_1 = timeit(first_order, number=1)
    print('FIRST ORDER TIME: {}'.format(time_1))
    time_2 = timeit(second_order, number=1)
    print('SECOND ORDER TIME: {}'.format(time_2))
    print('\n')
    res['1st order scheme'].append(time_1)
    res['2nd order scheme'].append(time_2)

    json_obj = json.dumps(res)
    f = open("misc-xp/results/sink_time.json".format(n), "w")
    f.write(json_obj)
    f.close()
