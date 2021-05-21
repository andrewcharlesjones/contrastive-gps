import tqdm
import math
import torch
import gpytorch
from matplotlib import pyplot as plt


import urllib.request
import os
from scipy.io import loadmat
from math import floor
import numpy as np


# this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)


if not smoke_test and not os.path.isfile('../elevators.mat'):
	print('Downloading \'elevators\' UCI dataset...')
	urllib.request.urlretrieve('https://drive.google.com/uc?export=download&id=1jhWL3YUHvXIaftia4qeAyDwVxo6j1alk', '../elevators.mat')


if smoke_test:  # this is for running the notebook in our testing framework
	X, y = torch.randn(1000, 3), torch.randn(1000)
else:
	# import ipdb; ipdb.set_trace()
	data = torch.Tensor(loadmat('../elevators.mat')['data'])
	X = data[:, :-1]
	X = X - X.min(0)[0]
	X = 2 * (X / X.max(0)[0]) - 1
	y = data[:, -1]

n = 2000
X = np.random.uniform(low=-1, high=1, size=(n, 1))
X[n//2:] += 5
y1 = (X[:n//2] * 1.5).squeeze()
y2 = (X[n//2:] * -1.5).squeeze()
y = np.concatenate([y1, y2]) + np.random.normal(scale=0.1, size=n)
X, y = torch.Tensor(X), torch.Tensor(y)

# plt.scatter(X[:, 0], y)
# plt.show()


train_n = int(floor(0.8 * len(X)))
train_x = X[:train_n, :].contiguous()
train_y = y[:train_n].contiguous()

test_x = X[train_n:, :].contiguous()
test_y = y[train_n:].contiguous()

if torch.cuda.is_available():
	train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()

# import ipdb; ipdb.set_trace()
from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

test_dataset = TensorDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=False)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

inducing_points = train_x[:10, :]
model = GPModel(inducing_points=inducing_points)
likelihood = gpytorch.likelihoods.GaussianLikelihood()

if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()


num_epochs = 1 if smoke_test else 4


model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.01)

# Our loss object. We're using the VariationalELBO
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))


epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
for i in epochs_iter:
    # Within each iteration, we will go over each minibatch of data
    minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
    for x_batch, y_batch in minibatch_iter:
        optimizer.zero_grad()
        # output = model(x_batch)
        # import ipdb; ipdb.set_trace()
        loss = -mll(model(train_x[:n//2]), train_y[:n//2]) + mll(model(train_x[n//2:]), train_y[n//2:])
        minibatch_iter.set_postfix(loss=loss.item())
        loss.backward()
        optimizer.step()


# inducing_points = model._modules['variational_strategy'].__dict__['_parameters']['inducing_points']


model.eval()
likelihood.eval()
means = torch.tensor([0.])
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        preds = model(x_batch)
        means = torch.cat([means, preds.mean.cpu()])
means = means[1:]

plt.scatter(test_x[:, 0], means, label="preds")
plt.scatter(X[:, 0], y, label="data")
plt.legend()
plt.show()

import ipdb; ipdb.set_trace()




