import numpy as np
from models import SSNB
from scipy.stats import ortho_group
from utils import semi_dual
import json
np.random.seed(0)

d = 8
l_range = [0.2, 0.5, 0.7, 0.9]
L_range = [0.3, 0.5, 0.7, 0.9, 1.3]

hyperparams = []
res = {}
for l in l_range:
    for L in L_range:
        if l<L:
            hyperparams.append((l, L))
            key = '(l, L) = {}'.format((l, L))
            res[key] = {'test_semi_dual': [], 'eval_l2_error_st': []}

n = 1024
n_trials = 10

lambd = 0.25

# Generate affine maps
orth_mat = ortho_group.rvs(d)
diag = np.diag(np.random.rand(d) + lambd)
rot = orth_mat@diag@orth_mat.T
inv_rot = np.linalg.inv(rot)
shift = np.random.randn(d)

def transport_map(x):
    return (rot@x.T + shift[:, None]).T
def inv_map(x):
    return (inv_rot@(x.T - shift[:, None])).T

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

    for (l, L) in hyperparams:

        key = '(l, L) = {}'.format((l, L))
        print(key)
        print('\n')

        ssnb = SSNB(mu_train, l, L)
        ssnb.train(nu_train, 10)

        # Compute test semi primal
        f = lambda x: ssnb.f_grad_f(x)[0]
        grad = lambda x: ssnb.f_grad_f(x)[1]
        semi_dual_test = semi_dual(mu_test, nu_test, f, grad, 1/(2*L), T=200)
        print('Semi Dual test: {}'.format(semi_dual_test))
        res[key]['test_semi_dual'].append(semi_dual_test.item())

        # Compute L2 error source to target on eval set
        map_st_error = ((grad(mu_eval) - transport_map(mu_eval))**2).sum(1).mean()
        res[key]['eval_l2_error_st'].append(map_st_error.item())
        print('L2 error source/target: {}'.format(map_st_error))
        print('\n')

    print('###################################################################')
    print('\n')

    json_obj = json.dumps(res)
    f = open("Synth-XP/Quad/results/ssnb.json", "w")
    f.write(json_obj)
    f.close()
