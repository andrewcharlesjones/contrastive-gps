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
        lower = np.zeros(X.shape[0])
        upper = np.zeros(X.shape[0])
        bg_idx, fg_idx, fg_only_idx = np.where(groups == 0)[0], np.where(groups == 1)[0], np.where(groups == 2)[0]
        Xtest_bg, Xtest_fg, Xtest_fg_only = torch.from_numpy(X[bg_idx]), torch.from_numpy(X[fg_idx]), torch.from_numpy(X[fg_only_idx])

        with torch.no_grad():
            preds_bg = self.likelihood(self.model_bg(Xtest_bg.double()))
            preds_fg = self.likelihood(self.model_fg(Xtest_fg.double()))
            preds_fg_only = self.likelihood(self.model_fg_only(Xtest_fg_only.double()))

        # Means
        preds[bg_idx] = preds_bg.mean.numpy()
        preds[fg_idx] = preds_fg.mean.numpy()
        preds[fg_only_idx] = preds_fg_only.mean.numpy()

        # Confidence regions
        conf_region_bg = preds_bg.confidence_region()
        lower_bg, upper_bg = conf_region_bg[0].numpy(), conf_region_bg[1].numpy()
        lower[bg_idx] = lower_bg
        upper[bg_idx] = upper_bg
        conf_region_fg = preds_fg.confidence_region()
        lower_fg, upper_fg = conf_region_fg[0].numpy(), conf_region_fg[1].numpy()
        lower[fg_idx] = lower_fg
        upper[fg_idx] = upper_fg
        conf_region_fg_only = preds_fg_only.confidence_region()
        lower_fg_only, upper_fg_only = conf_region_fg_only[0].numpy(), conf_region_fg_only[1].numpy()
        lower[fg_only_idx] = lower_fg_only
        upper[fg_only_idx] = upper_fg_only
        return preds, lower, upper


if __name__ == "__main__":

    import matplotlib
    font = {"size": 25}
    matplotlib.rc("font", **font)
    matplotlib.rcParams["text.usetex"] = True

    n = 100
    p = 1
    X = np.random.uniform(-5, 5, size=(n, p))
    groups = np.random.binomial(n=1, p=0.5, size=n)


    def bg_fn(x):
        return np.sin(x)
    def fg_fn(x):
        return np.sin(x * 3)
    y = (
        bg_fn(X).squeeze()
        + fg_fn(X).squeeze() * groups
        + np.random.normal(scale=0.1, size=n)
    )
    model = ContrastiveGaussianProcessRegression(
        num_mixtures_shared=1, num_mixtures_fg=1
    )
    model.fit(X, y, groups)

    n_test = 201
    X_test = np.linspace(-6, 6, n_test)
    preds_bg, lower_bg, upper_bg = model.predict(X_test, np.zeros(n_test))
    preds_fg_only, lower_fg_only, upper_fg_only = model.predict(X_test, 2*np.ones(n_test))
    preds_fg, lower_fg, upper_fg = model.predict(X_test, np.ones(n_test))

    X_bg, y_bg = X[groups == 0], y[groups == 0]
    X_fg, y_fg = X[groups == 1], y[groups == 1]

    plt.figure(figsize=(42, 14))

    ## True functions
    plt.subplot(321)
    plt.plot(X_test, bg_fn(X_test), linewidth=3)
    plt.ylim([-2, 2])
    plt.title(r"$f(x) = \sin(x)$")
    plt.subplot(323)
    plt.plot(X_test, fg_fn(X_test), linewidth=3)
    plt.ylim([-2, 2])
    plt.title(r"$f(x) = \sin(3x)$")
    plt.subplot(325)
    plt.plot(X_test, bg_fn(X_test) + fg_fn(X_test), linewidth=3)
    plt.ylim([-2, 2])
    plt.title(r"$f(x) = \sin(x) + \sin(3x)$")

    plt.subplot(322)
    plt.plot(X_test, preds_bg)
    plt.fill_between(X_test, lower_bg, upper_bg, alpha=0.5)
    plt.scatter(X_bg, y_bg, label="BG")
    plt.scatter(X_fg, y_fg, label="FG")
    plt.legend()
    plt.ylim([-2, 2])
    plt.title(r"$k(x, x^\prime) = \sum\limits_{q=1}^{Q_s} k_q(x, x^\prime)$" + " (predictions with shared kernels only)")
    plt.subplot(324)
    plt.plot(X_test, preds_fg_only)
    plt.fill_between(X_test, lower_fg_only, upper_fg_only, alpha=0.5)
    plt.scatter(X_bg, y_bg, label="BG")
    plt.scatter(X_fg, y_fg, label="FG")
    plt.legend()
    plt.ylim([-2, 2])
    plt.title(r"$k(x, x^\prime) = \sum\limits_{q=1}^{Q_f} k_q(x, x^\prime)$" + " (predictions with FG kernels only)")
    plt.subplot(326)
    plt.plot(X_test, preds_fg)
    plt.fill_between(X_test, lower_fg, upper_fg, alpha=0.5)
    plt.scatter(X_bg, y_bg, label="BG")
    plt.scatter(X_fg, y_fg, label="FG")
    plt.ylim([-2, 2])
    plt.title(r"$k(x, x^\prime) = \sum\limits_{q=1}^{Q_s} k_q(x, x^\prime) + \sum\limits_{q=1}^{Q_f} k_q(x, x^\prime)$" + " (predictions with FG+shared kernels)")
    plt.legend()
    plt.suptitle(r"$f(x_b) = \sin(x_b)$" + "\n" + r"$f(x_f) = \sin(x_f) + \sin(3x_f)$")
    plt.tight_layout()
    plt.savefig("../plots/sinusoidal_contrastive_gp.png")
    plt.close()
    # plt.show()
    # import ipdb
    # ipdb.set_trace()
