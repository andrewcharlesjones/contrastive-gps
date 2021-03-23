import gpflow
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf

from gpflow.utilities import ops, print_summary
from gpflow.config import set_default_float, default_float, set_default_summary_fmt
from gpflow.ci_utils import ci_niter

set_default_float(np.float64)


def sinusoid(x, scale=3, shift=0):
    return np.cos(scale * 2 * np.pi * (x[..., 0]) + shift)


SCALE_SHARED = 0.5
SCALE_FG = 1
SHIFT_FG = 0.25 * np.pi
SHIFT_SHARED = 0


def generate_1d_contrastive_data(num_training_points, observation_noise_variance):
    """Generate noisy sinusoidal observations at a random set of points.

    Returns:
       observation_index_points, observations
    """

    ### Background data
    index_points_bg = np.random.uniform(-2.0, 2.0, (num_training_points, 1))
    index_points_bg = index_points_bg.astype(np.float64)

    # y = f(x) + noise
    observations_bg = sinusoid(index_points_bg, scale=SCALE_SHARED) + np.random.normal(
        loc=0, scale=np.sqrt(observation_noise_variance), size=(num_training_points)
    )

    ### Foreground data
    index_points_fg1 = np.random.uniform(-2.0, 2.0, (num_training_points // 2, 1))
    index_points_fg1 = index_points_fg1.astype(np.float64)

    observations_fg1 = (
        sinusoid(index_points_fg1, scale=SCALE_SHARED, shift=SHIFT_SHARED)
        + sinusoid(index_points_fg1, scale=SCALE_FG, shift=SHIFT_FG)
        + np.random.normal(
            loc=0,
            scale=np.sqrt(observation_noise_variance),
            size=(num_training_points // 2),
        )
    )

    index_points_fg2 = np.random.uniform(-2.0, 2.0, (num_training_points // 2, 1))
    index_points_fg2 = index_points_fg2.astype(np.float64)

    observations_fg2 = (
        sinusoid(index_points_fg2, scale=SCALE_FG, shift=SHIFT_FG)
        + sinusoid(index_points_fg2, scale=SCALE_SHARED, shift=SHIFT_SHARED)
        + np.random.normal(
            loc=0,
            scale=np.sqrt(observation_noise_variance),
            size=(num_training_points // 2),
        )
    )

    index_points_fg = np.concatenate([index_points_fg1, index_points_fg2])
    observations_fg = np.concatenate([observations_fg1, observations_fg2])

    Y_bg = np.hstack([index_points_bg, np.expand_dims(observations_bg, 1)])
    Y_fg = np.hstack([index_points_fg, np.expand_dims(observations_fg, 1)])

    return Y_bg, Y_fg


# Generate training data with a known noise level (we'll later try to recover
# this value from the data).
NUM_TRAINING_POINTS = 100
Y_bg, Y_fg = generate_1d_contrastive_data(
    num_training_points=NUM_TRAINING_POINTS, observation_noise_variance=0.01
)

# plt.scatter(Y_bg[:, 0], Y_bg[:, 1])
# plt.scatter(Y_fg[:, 0], Y_fg[:, 1])
# plt.show()


Y = tf.convert_to_tensor(Y_fg, dtype=default_float())


print(
    "Number of points: {} and Number of dimensions: {}".format(Y.shape[0], Y.shape[1])
)

latent_dim = 2  # number of latent dimensions
num_inducing = 20  # number of inducing pts
num_data = Y.shape[0]  # number of data points

X_mean_init = ops.pca_reduce(Y, latent_dim)
# X_mean_init = tf.ones((num_data, latent_dim), dtype=default_float())
X_var_init = tf.ones((num_data, latent_dim), dtype=default_float())


np.random.seed(1)  # for reproducibility
inducing_variable = tf.convert_to_tensor(
    np.random.permutation(X_mean_init.numpy())[:num_inducing], dtype=default_float()
)

# lengthscales = tf.convert_to_tensor([1.0] * latent_dim, dtype=default_float())
# kernel = gpflow.kernels.RBF(lengthscales=lengthscales)
kernel = gpflow.kernels.Cosine()

gplvm = gpflow.models.BayesianGPLVM(
    Y,
    X_data_mean=X_mean_init,
    X_data_var=X_var_init,
    kernel=kernel,
    inducing_variable=inducing_variable,
)
# Instead of passing an inducing_variable directly, we can also set the num_inducing_variables argument to an integer, which will randomly pick from the data.

gplvm.likelihood.variance.assign(0.01)


opt = gpflow.optimizers.Scipy()
maxiter = ci_niter(1000)
_ = opt.minimize(
    gplvm.training_loss,
    method="BFGS",
    variables=gplvm.trainable_variables,
    options=dict(maxiter=maxiter),
)

print_summary(gplvm)


X_pca = ops.pca_reduce(Y, latent_dim).numpy()
gplvm_X_mean = gplvm.X_data_mean.numpy()

# plt.subplot(131)
# plt.scatter(Y[:, 0], gplvm_X_mean)
# plt.subplot(132)
# plt.scatter(Y[:, 1], gplvm_X_mean)
# plt.subplot(133)
# plt.scatter(Y_bg[:, 0], Y_bg[:, 1], c=gplvm_X_mean)
# plt.show()

plt.subplot(121)
plt.scatter(gplvm_X_mean[:, 0], gplvm_X_mean[:, 1])
plt.subplot(122)
plt.scatter(Y_bg[:, 0], Y_bg[:, 1])
plt.show()


import ipdb

ipdb.set_trace()
