import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tf.enable_v2_behavior()


class ContrastiveGaussianProcessRegression:
    def __init__(self, kernel_shared, kernel_fg):
        self.kernel_shared = kernel_shared
        self.kernel_fg = kernel_fg

    def fit(self, X, y, groups, noise_variance=1):
        self.X_train = X
        self.y_train = y
        self.groups_train = groups
        self.noise_variance = noise_variance

    def predict(self, X, groups):
        self.groups_test = groups

        # Split fg and bg
        train_bg_idx = np.argwhere(self.groups_train == 0).squeeze()
        train_fg_idx = np.argwhere(self.groups_train == 1).squeeze()
        test_bg_idx = np.argwhere(self.groups_test == 0).squeeze()
        test_fg_idx = np.argwhere(self.groups_test == 1).squeeze()
        X_train_bg = self.X_train[train_bg_idx, :]
        X_train_fg = self.X_train[train_fg_idx, :]
        y_train_bg = self.y_train[train_bg_idx]
        y_train_fg = self.y_train[train_fg_idx]

        X_test_bg = X[test_bg_idx, :]
        X_test_fg = X[test_fg_idx, :]

        preds = np.zeros(X.shape[0])

        # Background model
        gprm_bg = tfd.GaussianProcessRegressionModel(
            mean_fn=None,
            kernel=self.kernel_shared,
            index_points=X_test_bg,
            observation_index_points=X_train_bg,
            observations=y_train_bg,
            observation_noise_variance=self.noise_variance,
            predictive_noise_variance=0.0,
        )

        # Foreground model
        gprm_fg = tfd.GaussianProcessRegressionModel(
            mean_fn=None,
            kernel=self.kernel_shared + self.kernel_fg,
            index_points=X_test_fg,
            observation_index_points=X_train_fg,
            observations=y_train_fg,
            observation_noise_variance=self.noise_variance,
            predictive_noise_variance=0.0,
        )

        preds[test_bg_idx] = np.array(gprm_bg.mean())
        preds[test_fg_idx] = np.array(gprm_fg.mean())

        return preds
