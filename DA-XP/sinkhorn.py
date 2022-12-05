import os
from scipy.io import loadmat
import numpy as np
import torch
cuda = torch.cuda.is_available()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from scipy.optimize import minimize
import json
from models import SINKHORN
from utils import semi_dual

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
hyperparams =  [0.5, 0.1, 0.05, 0.01, 0.005]

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

            for hyp in hyperparams:

                print('EPSILON = {}'.format(hyp))

                sink = SINKHORN(X_s_train_tensor, X_t_train_tensor, hyp)
                sink.solve_dual(T=10000000, tol=1e-5)

                # Compute test semi primal
                f = lambda x: sink.f(x) + (reg/2)*(x**2).sum(-1)
                grad_f = lambda x: sink.grad_f(x) + reg*x
                if cuda:
                    hess_f = lambda x: sink.hess_f(x) + reg*torch.eye(16).cuda()[None, :]
                else:
                    hess_f = lambda x: sink.hess_f(x) + reg*torch.eye(16)[None, :]

                sd = semi_dual(X_s_test_tensor, X_t_test_tensor, f, grad_f, 100.0,
                              hess_f, T=1000000)

                res_dict[key]['test_sd'].append(sd.item())
                print('Semi Dual Test: {}'.format(sd))

                if cuda:
                    push = grad_f(X_s_tensor).cpu().numpy().astype('double')
                else:
                    push = grad_f(X_s_tensor).numpy().astype('double')

                knn.fit(push, Y_s)
                acc = accuracy_score(knn.predict(X_t), Y_t)

                res_dict[key]['accuracy'].append(acc)
                print('Accuracy: {}'.format(acc))

            json_obj = json.dumps(res_dict)
            f = open("DA-XP/results/sinkhorn.json", "w")
            f.write(json_obj)
            f.close()
