import os
from scipy.io import loadmat
import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from utils import semi_dual
import json
from models import SSNB

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

l_range = [0.2, 0.5, 0.7, 0.9]
L_range = [0.3, 0.5, 0.7, 0.9, 1.3]

hyperparams = []
for l in l_range:
    for L in L_range:
        if l<L:
            hyperparams.append((l, L))

res_dict = {}

knn = KNeighborsClassifier(n_neighbors=1)
pca = PCA(n_components=16)

for i in range(len(ds)):
    for j in range(len(ds)):
        #if i != j and names[i]!='Amazon' and (names[i], names[j])!=('Caltech', 'Amazon'):
        #if names[i]=='Dslr':

        key = '{}/{}'.format(names[i], names[j])
        res_dict[key]= {'test_sd': [], 'accuracy':[]}

        print('###########################################################')
        print('Source/Target: {}'.format(key))
        print('\n')

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

        for hyp in hyperparams:

            print('(l, L) = {}'.format(hyp))

            ssnb = SSNB(X_s_train, hyp[0], hyp[1])
            ssnb.train(X_t_train, 10, socp_tol=1e-5)

            # Compute test semi primal
            f = lambda x: ssnb.f_grad_f(x)[0]
            grad = lambda x: ssnb.f_grad_f(x)[1]
            sd = semi_dual(X_s_test, X_t_test, f, grad, 1/(2*L), T=200)

            res_dict[key]['test_sd'].append(sd.item())
            print('Semi Dual Test: {}'.format(sd))

            push = grad(X_s)
            knn.fit(push, Y_s)
            acc = accuracy_score(knn.predict(X_t), Y_t)

            res_dict[key]['accuracy'].append(acc)
            print('Accuracy: {}'.format(acc))

        json_obj = json.dumps(res_dict)
        f = open("DA-XP/results/ssnb.json", "w")
        f.write(json_obj)
        f.close()
