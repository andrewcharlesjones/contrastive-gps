import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
from itertools import chain


class ContrastiveGaussianProcessRegression:
    def __init__(self, num_mixtures_shared=5, num_mixtures_fg=5):
        self.num_mixtures_shared = num_mixtures_shared
        self.num_mixtures_fg = num_mixtures_fg
        self.covar_module_shared = gpytorch.kernels.SpectralMixtureKernel(
            num_mixtures=num_mixtures_shared
        )
        self.covar_module_fg = gpytorch.kernels.SpectralMixtureKernel(
            num_mixtures=num_mixtures_fg
        )
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    class BGModel(gpytorch.models.ExactGP):
        def __init__(self, X_bg, y_bg, likelihood, covar_module):
            super(ContrastiveGaussianProcessRegression.BGModel, self).__init__(
                X_bg, y_bg, likelihood
            )
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = covar_module

            # TODO initialize shared and fg kernels from data separately
            # self.covar_module.initialize_from_data(train_x, train_y)

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    class FGModel(gpytorch.models.ExactGP):
        def __init__(self, X_fg, y_fg, likelihood, covar_module):
            super(ContrastiveGaussianProcessRegression.FGModel, self).__init__(
                X_fg, y_fg, likelihood
            )
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = covar_module

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    class FGOnlyModel(gpytorch.models.ExactGP):
        def __init__(self, X_fg, y_fg, likelihood, covar_module):
            super(ContrastiveGaussianProcessRegression.FGOnlyModel, self).__init__(
                X_fg, y_fg, likelihood
            )
            self.covar_module = covar_module

        def forward(self, x):
            mean_x = torch.zeros(x.shape[0])
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def _split_data(self, X, y, groups):
        X_bg, X_fg = X[groups == 0], X[groups == 1]
        y_bg, y_fg = y[groups == 0], y[groups == 1]
        self.X_bg, self.X_fg = torch.from_numpy(X_bg), torch.from_numpy(X_fg)
        self.y_bg, self.y_fg = torch.from_numpy(y_bg), torch.from_numpy(y_fg)

    def fit(self, X, y, groups, training_iter=100):
        # Split into foreground and background
        self._split_data(X, y, groups)

        # Define models
        self.model_bg = self.BGModel(
            self.X_bg, self.y_bg, self.likelihood, self.covar_module_shared
        )
        self.model_fg = self.FGModel(
            self.X_fg,
            self.y_fg,
            self.likelihood,
            self.covar_module_shared + self.covar_module_fg,
        )
        self.model_fg_only = self.FGOnlyModel(
            self.X_fg, self.y_fg, self.likelihood, self.covar_module_fg
        )

        # Find optimal model hyperparameters
        self.model_bg.train()
        self.model_fg.train()
        self.model_fg_only.train()
        self.likelihood.train()

        # Use the adam optimizer
        self.optimizer = torch.optim.Adam(
            chain(
                self.model_fg.parameters(),
                self.model_bg.parameters(),
                self.covar_module_shared.parameters(),
                self.covar_module_fg.parameters(),
            ),
            lr=0.1,
        )

        # "Loss" for GPs - the marginal log likelihood
        self.mll_bg = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.model_bg
        )
        self.mll_fg = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.model_fg
        )
        self.mll_fg_only = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.model_fg_only
        )

        for i in range(training_iter):

            self.optimizer.zero_grad()
            output_bg = self.model_bg(self.X_bg)
            output_fg = self.model_fg(self.X_fg)
            loss = -self.mll_bg(output_bg, self.y_bg) - self.mll_fg(
                output_fg, self.y_fg
            )
            loss.backward()

            if (i + 1) % 10 == 0:
                print("Iter %d/%d - Loss: %.3f" % (i + 1, training_iter, loss.item()))

            self.optimizer.step()

    def predict(self, X, groups):
        self.mll_bg.eval()
        self.mll_fg.eval()
        self.mll_fg_only.eval()
        self.likelihood.eval()

        preds = np.zeros(X.shape[0])
        bg_idx, fg_idx = np.where(groups == 0)[0], np.where(groups == 1)[0]
        Xtest_bg, Xtest_fg = torch.from_numpy(X[bg_idx]), torch.from_numpy(X[fg_idx])

        with torch.no_grad():
            preds_bg = self.likelihood(self.model_bg(Xtest_bg.double()))
            preds_fg = self.likelihood(self.model_fg(Xtest_fg.double()))

        preds[bg_idx] = preds_bg.mean.numpy()
        preds[fg_idx] = preds_fg.mean.numpy()
        return preds


if __name__ == "__main__":
    n = 100
    p = 1
    X = np.random.uniform(-5, 5, size=(n, p))
    groups = np.random.binomial(n=1, p=0.5, size=n)
    # beta_shared = np.random.normal(size=p)
    # beta_fg = np.random.normal(size=p)
    # y = X @ beta_shared + (X @ beta_fg) * groups
    y = (
        np.sin(X).squeeze()
        + np.sin(X * 3).squeeze() * groups
        + np.random.normal(scale=0.1, size=n)
    )
    model = ContrastiveGaussianProcessRegression(
        num_mixtures_shared=1, num_mixtures_fg=1
    )
    model.fit(X, y, groups)

    n_test = 201
    X_test = np.linspace(-6, 6, n_test)
    groups_test = np.random.binomial(n=1, p=0.5, size=n_test)
    preds = model.predict(X_test, groups_test)
    import ipdb

    ipdb.set_trace()
