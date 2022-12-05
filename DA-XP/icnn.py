import os
from scipy.io import loadmat
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from scipy.optimize import minimize
import itertools
import json
import torch
cuda = torch.cuda.is_available()
from models import ICNNOT
from tqdm import tqdm
from torch.autograd.functional import jacobian
state = np.random.RandomState(0)

# Load data
path_to_data = os.path.join(os.getcwd(), 'DA-XP/data')
am = loadmat(os.path.join(path_to_data, 'amazon.mat'))
cal = loadmat(os.path.join(path_to_data, 'caltech.mat'))
dslr = loadmat(os.path.join(path_to_data, 'dslr.mat'))
web = loadmat(os.path.join(path_to_data, 'webcam.mat'))

ds = [am, cal, dslr, web]
names = ['Amazon', 'Caltech', 'Dslr', 'Web']
dtype = torch.float32
reg = 1e-3
list_n_neurons = [64, 128, 256]
list_lambd_cvx = [0.0, 0.001, 0.01, 0.1]
list_lambd_mean = [0.0, 0.001, 0.01, 0.1]

res_dict = {}

knn = KNeighborsClassifier(n_neighbors=1)
pca = PCA(n_components=16)

for i in range(len(ds)):
    for j in range(len(ds)):
        if i != j:

            key = '{}/{}'.format(names[i], names[j])

            print('###########################################################')
            print('Source/Target: {}'.format(key))
            print('\n')

            res_dict[key]= {'test_sd': [], 'accuracy':[]}

            source = ds[i]
            target = ds[j]

            n = len(source['labels'])

            data = np.concatenate([source['feas'], target['feas']])
            X = pca.fit_transform(data)

            X -= X.min()
            X /= X.max()

            X_s = X[:n, :]
            X_t = X[n:, :]
            Y_s = source['labels'].flatten()
            Y_t = target['labels'].flatten()


            X_s_train, X_s_test = train_test_split(X_s, test_size=0.3,
                                                    random_state = state)

            X_t_train, X_t_test = train_test_split(X_t, test_size=0.3,
                                                    random_state = state)

            X_s_tensor = torch.tensor(X_s, dtype=dtype).contiguous()
            X_t_tensor = torch.tensor(X_t, dtype=dtype).contiguous()
            X_s_train_tensor = torch.tensor(X_s_train, dtype=dtype).contiguous()
            X_t_train_tensor = torch.tensor(X_t_train, dtype=dtype).contiguous()
            X_s_test_tensor = torch.tensor(X_s_test, dtype=dtype).contiguous()
            X_t_test_tensor = torch.tensor(X_t_test, dtype=dtype).contiguous()

            if cuda:
                X_s_tensor = X_s_tensor.cuda()
                X_t_tensor = X_t_tensor.cuda()
                X_s_train_tensor = X_s_train_tensor.cuda()
                X_t_train_tensor = X_t_train_tensor.cuda()
                X_s_test_tensor = X_s_test_tensor.cuda()
                X_t_test_tensor = X_t_test_tensor.cuda()

            for hyp in itertools.product(list_n_neurons, list_lambd_cvx, list_lambd_mean):

                print('(N NEURONS, LAMBD CVX, LAMBD MEAN) = {}'.format(hyp))

                icnnot = ICNNOT(n_neurons=hyp[0], lambd_cvx=hyp[1],
                                lambd_mean=hyp[2])
                icnnot.train(X_s_train_tensor, X_t_train_tensor)

                # Compute test semi primal
                f = lambda x: icnnot.convex_f(x)
                sum_f = lambda x: f(x).sum()
                grad = lambda x: jacobian(sum_f, x)

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
                for y in tqdm(X_t_test, 'Computing Fenchel transform ...'):

                    g = lambda x: f_numpy(x) + reg*(x**2).sum()/2 - (y*x).sum()
                    grad_g = lambda x: grad_numpy(x) + reg*x - y

                    prob = minimize(g, np.random.randn(16), jac=grad_g, tol=1e-3)
                    while not prob['success']:
                        prob = minimize(g, np.random.randn(16), jac=grad_g, tol=1e-3)
                    f_t.append(prob['x'])

                if cuda:
                    f_t = torch.tensor(f_t, dtype=dtype).cuda()
                else:
                    f_t = torch.tensor(f_t, dtype=dtype)
                mu_term = (f(X_s_test_tensor) + reg*(X_s_test_tensor**2).sum(1)/2).mean()
                nu_term = (f_t*X_t_test_tensor).sum(1).mean() - (f(f_t) + reg*(f_t**2).sum(1)/2).mean()
                sd = mu_term + nu_term

                res_dict[key]['test_sd'].append(sd.item())

                print('Semi Dual Test: {}'.format(sd))

                if cuda:
                    push = grad(X_s_tensor).cpu().numpy().astype('double')
                else:
                    push = grad(X_s_tensor).numpy().astype('double')
                knn.fit(push, Y_s)
                acc = accuracy_score(knn.predict(X_t), Y_t)

                res_dict[key]['accuracy'].append(acc)
                print('Accuracy: {}'.format(acc))
                print('\n')

            json_obj = json.dumps(res_dict)
            f = open("DA-XP/results/icnnot.json", "w")
            f.write(json_obj)
            f.close()
