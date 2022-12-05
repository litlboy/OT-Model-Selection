import torch
import torch.nn as nn
import numpy as np
from scipy.stats import truncnorm
from tqdm import tqdm
import sys

def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()

# Fenchel transform of genereic function f with 2nd order scheme
def fenchel_transform_newton(nu, grad_f, hess_f, T=50, tol=1e-6, self_conc=1.0):

    # Random start
    x = nu.clone()

    if nu.is_cuda:
        mask = torch.ones(len(x), dtype=bool).cuda()
        grad_y_f = torch.zeros(nu.shape, dtype=nu.dtype).cuda()
    else:
        mask = torch.ones(len(x), dtype=bool)
        grad_y_f = torch.zeros(nu.shape, dtype=nu.dtype)

    for t in tqdm(range(T), 'Computing conjugate ...'):

        grad_y_f[mask] = grad_f(x[mask])
        grad_norms = ((((nu - grad_y_f)**2).sum(1))**0.5)
        norm_grad_mean = grad_norms.mean()

        mask[grad_norms<tol] = False

        if mask.sum() == 0 or norm_grad_mean < tol:
            return x

        else:

            grad_y_g = nu[mask] - grad_y_f[mask]
            hess = hess_f(x[mask])

            p = torch.linalg.solve(hess, grad_y_g)
            norms_p = (p**2).sum(1)**0.5
            alpha = self_conc*norms_p
            x[mask] += p*torch.log(1+alpha[:, None])/alpha[:, None]

    print('Max number of iterations reached.')
    print('Average gradient norm: {}'.format(norm_grad_mean))
    return x

# Fenchel transform of genereic function f with 1st order scheme
def fenchel_transform_gd(nu, grad_f, T=50, tol=1e-5, lr=1.0):

    if torch.is_tensor(nu):

        x = nu.clone()
        if nu.is_cuda:
            mask = torch.ones(len(x), dtype=bool).cuda()
            grad_y_f = torch.zeros(nu.shape).cuda()

        else:
            mask = torch.ones(len(x), dtype=bool)
            grad_y_f = torch.zeros(nu.shape)

    else:
        x = nu.copy()
        mask = np.ones(len(x), dtype=bool)
        grad_y_f = np.zeros(nu.shape)

    for t in tqdm(range(T), 'Computing conjugate ...'):

        grad_y_f[mask] = grad_f(x[mask])
        grad_norms = ((((nu - grad_y_f)**2).sum(1))**0.5)
        norm_grad_mean = grad_norms.mean()

        mask[grad_norms<tol] = False

        if t%1000 == 0:
            print(grad_norms[mask])

        if mask.sum() == 0 or norm_grad_mean < tol:
            return x

        else:

            grad_y_g = nu[mask] - grad_y_f[mask]
            x[mask] += lr*grad_y_g

    print('Max number of iterations reached.')
    print('Average gradient norm: {}'.format(norm_grad_mean))
    return x

def semi_dual(mu, nu, f, grad_f, lr, hess_f=None, tol=1e-4, T=50):

    mu_term = f(mu).mean()

    if hess_f is None:
        f_t = fenchel_transform_gd(nu, grad_f, lr=lr, T=T, tol=tol)

    else:
        f_t = fenchel_transform_newton(nu, grad_f, hess_f,
                                       self_conc=lr, T=T, tol=tol)

    nu_term = (f_t*nu).sum(1).mean() - f(f_t).mean()

    return mu_term + nu_term

def truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values

def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU(0.2)
    elif activation == 'celu':
        return nn.CELU()
    elif activation == 'selu':
        return nn.SELU()
    elif activation == 'softplus':
        return nn.Softplus()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError('activation [%s] is not found' % activation)


class ConvexLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):

        super(ConvexLinear, self).__init__(*kargs, **kwargs)

        if not hasattr(self.weight, 'be_positive'):
            self.weight.be_positive = 1.0

    def forward(self, input):

        out = nn.functional.linear(input, self.weight, self.bias)

        return out

class Simple_Feedforward_3Layer_ICNN_LastInp_Quadratic(nn.Module):

    def __init__(self, input_dim, hidden_dim, activation):

        super(Simple_Feedforward_3Layer_ICNN_LastInp_Quadratic, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        # x -> h_1
        self.fc1_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        #self.dense1_bn = nn.BatchNorm1d(self.hidden_dim)
        self.activ_1 = get_activation(self.activation)

        self.fc2_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.fc2_convex = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activ_2 = get_activation(self.activation)

        self.fc3_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.fc3_convex = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activ_3 = get_activation(self.activation)

        self.last_convex = ConvexLinear(self.hidden_dim, 1, bias=False)
        self.last_linear = nn.Linear(self.input_dim, 1, bias=True)

    def forward(self, input):

        x = self.activ_1(self.fc1_normal(input)).pow(2)

        x = self.activ_2(self.fc2_convex(x).add(self.fc2_normal(input)))

        x = self.activ_3(self.fc3_convex(x).add(self.fc3_normal(input)))

        x = self.last_convex(x).add(self.last_linear(input).pow(2))

        return x
