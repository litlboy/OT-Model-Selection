import torch
torch.cuda.is_available()
from pykeops.torch import LazyTensor
import numpy as np
import cvxpy as cp
import sys
import ot
import os
from tqdm import tqdm
from cvxopt import solvers, matrix, spdiag, spmatrix, sparse, log
solvers.options['show_progress'] = False
from utils import *
import mosek

# Sinkhorn model with LazyTensors to keep linear memory footprint
class SINKHORN(object):

    def __init__(self, mu, nu, epsilon):

        self.dtype = mu.dtype
        self.mu = mu
        self.nu = nu
        self.n = mu.shape[0]
        self.m = nu.shape[0]
        self.d = mu.shape[1]

        # Use symbolic tensors to scale
        self.mu_lazy = LazyTensor(mu[:, None, :])
        self.nu_lazy = LazyTensor(nu[None, :, :])
        mu_auto = (mu[:, :, None]@mu[:, None, :]).reshape(self.n, self.d**2)
        nu_auto = (nu[:, :, None]@nu[:, None, :]).reshape(self.m, self.d**2)
        self.mu_auto_l = LazyTensor(mu_auto[:, None, :])
        self.nu_auto_l = LazyTensor(nu_auto[None, :, :])

        # Regularization parameter
        self.eps = epsilon

        # Switch on cuda if available
        if torch.cuda.is_available():
            self.u = torch.zeros(self.n, dtype=self.dtype).cuda()
            self.v = torch.zeros(self.m, dtype=self.dtype).cuda()
            self.w_u = torch.ones(self.m, dtype=self.dtype).cuda()
            self.w_v = torch.ones(self.n, dtype=self.dtype).cuda()
        else:
            self.u = torch.zeros(self.n, dtype=self.dtype)
            self.v = torch.zeros(self.m, dtype=self.dtype)
            self.w_u = torch.ones(self.m, dtype=self.dtype)
            self.w_v = torch.ones(self.n, dtype=self.dtype)

    # Solve dual with T steps
    def solve_dual(self, T, tol=1e-3):

        gap_1 = False
        gap_2 = False

        if torch.cuda.is_available():
            log_m = torch.tensor([np.log(self.m)], dtype=self.dtype).cuda()
            log_n = torch.tensor([np.log(self.n)], dtype=self.dtype).cuda()

        else:
            log_m = LazyTensor(torch.tensor([np.log(self.m)], dtype=self.dtype)[None, :, None])
            log_n = LazyTensor(torch.tensor([np.log(self.n)], dtype=self.dtype)[:, None, None])

        c = ((self.mu_lazy - self.nu_lazy)**2).sum(-1)/2

        # Logsumexp updates
        for _ in tqdm(range(T), 'Solving dual ...'):

            # u step
            s_1 = ((LazyTensor(self.v[None, :, None]) - c)/self.eps - log_m).logsumexp(1).squeeze()
            if (s_1 + self.u/self.eps).abs().mean() < tol:
                gap_1 = True
            else:
                gap_1 = False
            self.u = -self.eps*s_1

            # v step
            s_2 = ((LazyTensor(self.u[:, None, None]) - c)/self.eps - log_n).logsumexp(0).squeeze()
            if (s_2 + self.v/self.eps).abs().mean() < tol:
                gap_2 = True
            else:
                gap_2 = False
            self.v = -self.eps*s_2

            if gap_1 and gap_2:
                print('Optim terminated')
                break

        self.shift_u = (self.v - (self.nu**2).sum(1)/2)/self.eps - torch.tensor([np.log(self.m)], dtype=self.dtype)
        self.shift_v = (self.u - (self.mu**2).sum(1)/2)/self.eps - torch.tensor([np.log(self.n)], dtype=self.dtype)

    # Bernier potentials, gradients, hessians
    def f(self, x):
        pairs = LazyTensor(x[:, None, :])*self.nu_lazy
        affine = pairs.sum(-1)/self.eps + self.shift_u[None, :, None]
        return self.eps*affine.logsumexp(1).squeeze()

    def g(self, y):
        pairs = LazyTensor(y[None, :, :])*self.mu_lazy
        affine = pairs.sum(-1)/self.eps + self.shift_v[:, None, None]
        return self.eps*affine.logsumexp(0).squeeze()

    def grad_f(self, x):
        pairs = LazyTensor(x[:, None, :])*self.nu_lazy
        affine = pairs.sum(-1)/self.eps + self.shift_u[None, :, None]
        max_affine = affine.max(1)
        kern = (affine - max_affine[:, None]).exp()
        return (self.nu_lazy*kern).sum(1)/kern.sum(1)

    def grad_g(self, y):
        pairs = LazyTensor(y[None, :, :])*self.mu_lazy
        affine = pairs.sum(-1)/self.eps + self.shift_v[:, None, None]
        max_affine = affine.max(0)
        kern = (affine - max_affine[None, :]).exp()
        return (self.mu_lazy*kern).sum(0)/kern.sum(0)

    def hess_f(self, x):
        pairs = LazyTensor(x[:, None, :])*self.nu_lazy
        affine = pairs.sum(-1)/self.eps + self.shift_u[None, :, None]
        max_affine = affine.max(1)
        kern = (affine - max_affine[:, None]).exp()
        denom = kern.sum(1)
        dom_term = (self.nu_auto_l*kern).sum(1)/denom
        nabla_f = (self.nu_lazy*kern).sum(1)/denom
        corr_term = nabla_f[:, :, None]@nabla_f[:, None, :]
        return (dom_term.reshape(x.shape[0], self.d, self.d) - corr_term)/self.eps

    def hess_g(self, y):
        pairs = LazyTensor(y[None, :, :])*self.mu_lazy
        affine = pairs.sum(-1)/self.eps + self.shift_v[:, None, None]
        max_affine = affine.max(0)
        kern = (affine - max_affine[None, :]).exp()
        denom = kern.sum(0)
        dom_term = (self.mu_auto_l*kern).sum(0)/denom
        nabla_g = (self.mu_lazy*kern).sum(0)/denom
        corr_term = nabla_g[:, :, None]@nabla_g[:, None, :]
        return (dom_term.reshape(y.shape[0], self.d, self.d) - corr_term)/self.eps

# Mosek Implementation of Smooth Strongly Convex Nearest Brenier model
class SSNB(object):

    def __init__(self, mu, l, L):

        self.l = l # strong convexity parameter
        self.L = L # smoothness parameter
        self.mu = mu
        self.n, self.d = mu.shape
        # Path for complied cone convexity constraint
        self.comp_train_path = 'compiled_ssnb/n={}_d={}_train.gz'.format(self.n,self.d)
        self.comp_grad_path = 'compiled_ssnb/n={}_d={}_grad.gz'.format(self.n,self.d)

        n_eq = self.n*(self.n - 1) # Number of constraints to enforce convexity
        n_var = self.n*(1+self.d) # Number of variables to optimize on (n values + n gradients)
        self.c = self.l/(self.L - self.l)

        self.h = matrix(0.0, (n_eq*(self.d+3),1))
        list_val = []
        list_idx_row = []
        list_idx_col = []
        count = 0
        # Build cone convexity constraint (does not change throughout the steps)
        # If f has values u = [u_1,...,  u_n] and gradients g = [g_1, ..., g_n]
        # on samples x = [x_1, ..., x_n], then f is interpolable by a l-strongly
        # convex, L-smooth function iff G [t, u , g] - h, where t is a slack
        # variable of size n_eq + 1, belongs to the cone
        # {0}x ... x{0} (n_eq times) x Kx...xK (n_eq+1 times) where K is the
        # rotated Lorentz cone of size d+2.
        for i in tqdm(range(self.n), 'Formatting cvxopt objects ...'):
            for j in range(self.n):
                if i != j:
                    # Completing G matrix
                    list_val += (self.c*(self.mu[j]-self.mu[i])).tolist()
                    list_idx_row += [count]*self.d
                    list_idx_col += range(i*self.d+n_eq+1, (i+1)*self.d+n_eq+1)
                    list_val += ((1+self.c)*(self.mu[i]-self.mu[j])).tolist()
                    list_idx_row += [count]*self.d
                    list_idx_col += range(j*self.d+n_eq+1, (j+1)*self.d+n_eq+1)
                    list_val += [-1.0, 1.0]
                    list_idx_row += [count]*2
                    list_idx_col += [1+i+n_eq+self.d*self.n, 1+j+n_eq+self.d*self.n]
                    list_val += [1.0]
                    list_idx_row += [count]
                    list_idx_col += [1+count]

                    # Completing Q_i matrices
                    list_val += [-1.0]
                    list_idx_col += [1+count]
                    list_idx_row += [n_eq + count*(self.d+2)]
                    val = np.ones(self.d)*(self.c/self.l)**0.5
                    list_val += np.concatenate([val, -val]).tolist()
                    list_idx_col += range(i*self.d+n_eq+1, (i+1)*self.d+n_eq+1)
                    list_idx_col += range(j*self.d+n_eq+1, (j+1)*self.d+n_eq+1)
                    offset = n_eq + count*(self.d+2)
                    list_idx_row += range(2 + offset, self.d + 2 + offset)
                    list_idx_row += range(2 + offset, self.d + 2 + offset)

                    # completing h
                    self.h[count] = -self.L*self.c*((self.mu[i]-self.mu[j])**2).sum()/2
                    self.h[n_eq + count*(self.d + 2) + 1] = 1.0
                    count += 1

        self.G = spmatrix(list_val, list_idx_row, list_idx_col,
                            size=(n_eq*(self.d+3), n_eq+1+n_var))

        self.lo_grad = matrix(0.0, (self.n+self.d+1, 1))
        self.lo_grad[self.n] = 1.0
        self.h_grad = matrix(0.0, (self.n*(self.d+3), 1))

        # To interpolate the potential f on some point x, solve the problem
        # \inf_{v, g} v under the constraint G_grad [u,v] - h_grad \in K where
        # K is a cone
        list_val = []
        list_idx_row = []
        list_idx_col = []
        for i in range(self.n):
            list_val.append(-1.0)
            list_idx_row.append(i*(self.d + 3))
            list_idx_col.append(i)
            list_val += (np.ones(self.d)/(self.L - self.l)**0.5).tolist()
            list_idx_row += np.arange(i*(self.d+3)+3, (i+1)*(self.d+3)).tolist()
            list_idx_col += np.arange(self.n + 1, self.n + self.d + 1).tolist()
            self.h_grad[i*(self.d + 3) + 1] = 1.0

        self.G_grad = spmatrix(list_val, list_idx_row, list_idx_col,
                            size=(self.n*(self.d+3), self.n+self.d+1))

    def train(self, nu, T, socp_tol=1e-8):

        n_eq = self.n*(self.n - 1)
        n_var = self.n*(1+self.d)
        idx_start = n_eq + 1
        idx_end = n_eq + 1 + self.d*self.n

        print('Starting alternate minimization')
        print('\n')
        print('############# STEP 0 #############')
        print('\n')
        print('Solving EMD ...')

        # Initialize OT couplings
        M = (self.mu**2).sum(1)[:, None] + (nu**2).sum(1)[None, :]
        M -= 2*self.mu@nu.T
        pi = ot.lp.emd(np.ones(self.mu.shape[0])/self.mu.shape[0],
                       np.ones(nu.shape[0])/nu.shape[0], M)

        # Training loop
        for t in range(T):

            # SOCP step

            # Linear objective
            c = (-pi@nu).ravel()
            lin_obj = matrix([0.0]*(n_eq+1+n_var), (n_eq+1+n_var, 1))
            lin_obj[0] = 1.0
            lin_obj[n_eq+1:len(c)+n_eq+1] = c
            pi_1 = pi.sum(1)
            diag = (pi_1[:, None]*np.ones(self.d)[None, :]).ravel()

            # SOCP constraint
            F = spmatrix(diag**0.5, np.arange(self.n*self.d),
                        np.arange(self.n*self.d), (n_var, n_var))
            a = spmatrix([-1.0], [0], [0], (2, n_eq+1))
            G = sparse([[a, spmatrix([], [], [], (n_var, n_eq+1))],
                        [spmatrix([], [], [], (2, n_var)), F]])
            h = matrix([0.0]*(n_var+2), (n_var+2,1))
            h[1] = 1.0

            # Check if the cone of convexity constraint has already be compiled
            compile = not os.path.exists(self.comp_train_path)

            # Solve SOCP of the form \inf_{x} c^\top x under the linear constraint
            # Gx - h \in K where K is the product of cones {0} x ... x {0} mp times,
            # R_+ \times R_+ ml times and L_1 x ... x L_q where q is the lenght
            # of mq and L_i is the rotated Lorentz cone of size mq[i]
            sol = self.socp(c=lin_obj, G=sparse([self.G, G]),
                        h=matrix([self.h, h]), mp=n_eq, ml=0,
                        mq=[self.d+2]*n_eq + [n_var+2], compile=compile,
                        tol=socp_tol)

            z = np.array(sol[idx_start:idx_end]).reshape(self.n, self.d)
            print('\n')
            print('############# STEP {} #############'.format(t+1))
            print('Solving EMD ...')
            # OT step
            M = (z**2).sum(1)[:, None] + (nu**2).sum(1)[None, :]
            M -= 2*z@nu.T
            pi = ot.lp.emd(np.ones(z.shape[0])/z.shape[0],
                           np.ones(nu.shape[0])/nu.shape[0], M)

        self.z = z
        self.u = np.array(sol[idx_end:])[:, 0]
        print('Estimated W2: {}'.format((pi*M).sum()))

    # Compute pointwise value and gradient of current potential estimate
    def f_grad_f(self, x):

        f = []
        grad_f = []

        for x_i in x:

            G, h = self.build_G_h(x_i)
            compile = not os.path.exists(self.comp_grad_path)
            mp = self.n
            ml = 0
            mq = [self.d + 3]*self.n
            sol = np.array(self.socp(self.lo_grad, G, h, mp, ml, mq,
                                    compile=compile, grad=True))[:, 0]
            f.append(sol[self.n])
            grad_f.append(sol[1+self.n:])

        return np.array(f), np.array(grad_f)

    def build_G_h(self, x):
        dx = x[None, :] - self.mu
        z_2 = (self.z**2).sum(1)
        dx_2 = (dx**2).sum(1)
        r = self.u+(dx*self.z).sum(1)*(1+self.c)+0.5*self.c*(dx_2*self.L+z_2/self.l)
        h = matrix([matrix(-r), self.h_grad])
        c = matrix(-1.0, (self.n, 1))
        A = matrix([c.T, matrix(-self.c*(dx + self.z/self.l)).T]).T
        A = sparse([spdiag(np.ones(self.n).tolist()), A.T]).T
        G = sparse([A, self.G_grad])
        return G, h

    # Mosek implemntation of an SOCP problem
    def socp(self, c, G, h, mp, ml, mq, compile=False, grad=False,
             tol=1e-8, verbose=False):

        with mosek.Env() as env:

            n = c.size[0]
            N = mp + ml + sum(mq)

            blc = list(-c)
            buc = list(-c)
            bkc = n*[ mosek.boundkey.fx ]

            c   = -h

            colptr, asub, acof = sparse([G.T]).CCS
            aptrb, aptre = colptr[:-1], colptr[1:]

            with env.Task(0,0) as task:

                if verbose:
                    task.set_Stream (mosek.streamtype.log, streamprinter)

                if grad:
                    comp_path = self.comp_grad_path
                else:
                    comp_path = self.comp_train_path

                if not compile:

                    task.readtask(comp_path)

                    # Reset value of objective
                    task.putcslice(0, c.size[0], list(c))

                    # Reset value of A
                    task.putacollist(list(range(G.size[0])), list(aptrb),
                                    list(aptre), list(asub), list(acof))

                    # Reset value of equality constraint
                    task.putconboundlist(list(range(n)), bkc, blc, buc)

                else:
                    bkx = mp*[ mosek.boundkey.fr ] + ml*[ mosek.boundkey.lo ] + sum(mq)*[ mosek.boundkey.fr ]
                    blx = mp*[ -0.0 ] + ml*[ 0.0 ] + sum(mq)*[ -0.0 ]
                    bux = N*[ +0.0 ]

                    task.inputdata (n,   # number of constraints
                                    N,   # number of variables
                                    list(c), # linear objective coefficients
                                    0.0, # objective fixed value
                                    list(aptrb),
                                    list(aptre),
                                    list(asub),
                                    list(acof),
                                    bkc,
                                    blc,
                                    buc,
                                    bkx,
                                    blx,
                                    bux)

                    task.putobjsense(mosek.objsense.maximize)

                    for k in tqdm(range(len(mq)), 'Compiling Cones ...'):
                        task.appendcone(mosek.conetype.rquad, 0.0,
                                        list(range(mp+ml+sum(mq[:k]),mp+ml+sum(mq[:k+1]))))

                    task.writetask(comp_path)

                if not grad:
                    print('Solving SOCP ...')

                task.putnadouparam("MSK_DPAR_INTPNT_TOL_REL_GAP",  tol)

                task.optimize()

                #task.solutionsummary (mosek.streamtype.msg);

                #solsta = task.getsolsta(mosek.soltype.itr)

                xu, xl = n*[0.0], n*[0.0]
                task.getsolutionslice(mosek.soltype.itr, mosek.solitem.slc, 0, n, xl)
                task.getsolutionslice(mosek.soltype.itr, mosek.solitem.suc, 0, n, xu)
                x = matrix(xu) - matrix(xl)

        return x

# Implementation of the Input Convex Neural Networks of Optimal Transport
# Adaptation of the original code: 'https://github.com/AmirTag/OT-ICNN'
class ICNNOT(object):

    def __init__(self, n_neurons=128, activation='softplus',
                lambd_cvx=0.01, lambd_mean=-1.0, n_inner=25):

        self.lambd_cvx = lambd_cvx
        self.lambd_mean = lambd_mean
        self.n_inner = n_inner
        self.n_neurons = n_neurons
        self.activation = activation

        self.convex_f = None
        self.convex_g = None
        self.optimizer_f = None
        self.optimizer_g = None
        self.cuda = False

        self.train_loader_mu = None
        self.train_loader_mu = None

    def train(self, mu, nu, epochs=50, batch_size=60, lr=1e-4):

        self.convex_f = Simple_Feedforward_3Layer_ICNN_LastInp_Quadratic(mu.shape[1], self.n_neurons, self.activation)
        self.convex_g = Simple_Feedforward_3Layer_ICNN_LastInp_Quadratic(mu.shape[1], self.n_neurons, self.activation)

        self.f_positive_params = []

        for p in list(self.convex_f.parameters()):
            if hasattr(p, 'be_positive'):
                self.f_positive_params.append(p)

            p.data = torch.from_numpy(truncated_normal(p.shape, threshold=1./np.sqrt(p.shape[1] if len(p.shape)>1 else p.shape[0]))).float()

        self.g_positive_params = []

        for p in list(self.convex_g.parameters()):
            if hasattr(p, 'be_positive'):
                self.g_positive_params.append(p)

            p.data = torch.from_numpy(truncated_normal(p.shape, threshold=1./np.sqrt(p.shape[1] if len(p.shape)>1 else p.shape[0]))).float()

        self.train_loader_mu = torch.utils.data.DataLoader(mu, batch_size=batch_size, shuffle=True)
        self.train_loader_nu = torch.utils.data.DataLoader(nu, batch_size=batch_size, shuffle=True)

        if mu.is_cuda:
            self.cuda = True
            self.convex_f.cuda()
            self.convex_g.cuda()

        self.optimizer_f = torch.optim.Adam(self.convex_f.parameters(), lr=lr, betas=(0.5, 0.99))
        self.optimizer_g = torch.optim.Adam(self.convex_g.parameters(), lr=lr, betas=(0.5, 0.99))

        for epoch in tqdm(range(1, epochs + 1), 'Optimizing ...'):
            self._step()

            if epoch % 2 == 0:

                self.optimizer_g.param_groups[0]['lr'] *= 0.5
                self.optimizer_f.param_groups[0]['lr'] *= 0.5

    def _step(self):

        self.convex_f.train()
        self.convex_g.train()


        for batch_idx, (mu_data, nu_data) in enumerate(zip(self.train_loader_mu,
                                                         self.train_loader_nu)):

            if self.cuda:

                mu_data = mu_data.cuda()
                nu_data = mu_data.cuda()

            nu_data.requires_grad = True

            self.optimizer_f.zero_grad()
            self.optimizer_g.zero_grad()

            loss_g_val = 0
            norm_g_parms_grad_full = 0

            for inner_iter in range(1, self.n_inner + 1):
                # First do a forward pass on y and compute grad_g_y
                # Then do a backward pass update on parameters of g

                self.optimizer_g.zero_grad()

                g_of_y = self.convex_g(nu_data).sum()

                grad_g_of_y = torch.autograd.grad(g_of_y, nu_data, create_graph=True)[0]

                f_grad_g_y = self.convex_f(grad_g_of_y).mean()

                loss_g = f_grad_g_y - torch.dot(grad_g_of_y.reshape(-1), nu_data.reshape(-1)) / nu_data.size(0)
                loss_g_val += loss_g.item()

                if self.lambd_mean > 0:

                    mean_difference_loss = self.lambd_mean * (mu_data.mean(0) - grad_g_of_y.mean(0)).pow(2).sum()
                    variance_difference_loss = self.lambd_mean * (mu_data.std(0) - grad_g_of_y.std(0)).pow(2).sum()

                    loss_g += mean_difference_loss + variance_difference_loss

                #print((mu_data.mean(0) - grad_g_of_y.mean(0)).pow(2).sum())

                loss_g.backward()

                g_params_grad_full = torch.cat([p.grad.reshape(-1).data.cpu() for p in list(self.convex_g.parameters())])
                norm_g_parms_grad_full += torch.norm(g_params_grad_full).item()

                if self.lambd_cvx > 0:
                    g_positive_constraint_loss = self.lambd_cvx*self._compute_constraint_loss(self.g_positive_params)
                    g_positive_constraint_loss.backward()

                self.optimizer_g.step()


                ## Maintaining the positive constraints on the convex_g_params
                if self.lambd_cvx == 0:
                    for p in self.g_positive_params:
                        p.data.copy_(torch.relu(p.data))


                ### Just for the last iteration keep the gradient on f intact
                ### otherwise need to do from scratch
                if inner_iter != self.n_inner:
                    self.optimizer_f.zero_grad()

            loss_g_val /= self.n_inner
            norm_g_parms_grad_full /= self.n_inner

            ## Flip the gradient sign for parameters in convex_f
            # because they are slow
            #for p in list(self.convex_f.parameters()):
            #    p.grad.copy_(-p.grad)

            nu_data.requires_grad = False
            grad_g_of_y = torch.autograd.functional.jacobian(lambda x: self.convex_g(x).sum(), nu_data)

            remaining_f_loss = self.convex_f(mu_data).mean() - self.convex_f(grad_g_of_y).mean()
            remaining_f_loss.backward()
            #print(remaining_f_loss)

            self.optimizer_f.step()


            # Maintain the "f" parameters positive
            for p in self.f_positive_params:
                p.data.copy_(torch.relu(p.data))


            #w_2_loss_value = loss_g_val-remaining_f_loss.item()+0.5*mu_data.pow(2).sum(dim=1).mean().item()+0.5*nu_data.pow(2).sum(dim=1).mean().item()
            #print(w_2_loss_value)

    def _compute_constraint_loss(self, list_of_params):

        loss_val = 0

        for p in list_of_params:
            loss_val += torch.relu(-p).pow(2).sum()
        return loss_val

    def nabla_f(self, x):
        sum_f = lambda x: self.convex_f(x).sum()
        return torch.autograd.functional.jacobian(sum_f, x)

    def hess_f(self, x):
        hess_tensor = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).cuda()
        for i, x_i in tqdm(enumerate(x), 'Computing hessian ...'):
            hess_tensor[i] = torch.autograd.functional.hessian(self.convex_f,
                                                                x_i[None, :])[0][:, 0, :]
        return hess_tensor
