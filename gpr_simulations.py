import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tf.enable_v2_behavior()

observation_index_points = np.random.uniform(-3., 3., (300, 1))



## Kernels
kernel1 = tfk.ExpSinSquared(np.float64(10.), np.float64(1.), np.float64(1.))
kernel2 = tfk.ExpSinSquared(np.float64(5.), np.float64(1.), np.float64(1.))
# kernel1 = tfk.ExponentiatedQuadratic(np.float64(1.), np.float64(.05))
# kernel2 = tfk.ExponentiatedQuadratic(np.float64(1.), np.float64(0.05))
# kernel1 = tfk.Linear()
# kernel2 = tfk.Linear()
kernel12 = kernel1 + kernel2

noise_variance = np.float64(0.1)

## GPs
gp1 = tfd.GaussianProcess(
      mean_fn=lambda x: np.float64(0.),
      # mean_fn=lambda x: np.float64(10) * x,
      kernel=kernel1,
      index_points=observation_index_points,
      observation_noise_variance=noise_variance)

gp2 = tfd.GaussianProcess(
      mean_fn=lambda x: np.float64(0.),
      kernel=kernel2,
      index_points=observation_index_points,
      observation_noise_variance=noise_variance)

gp12 = tfd.GaussianProcess(
      mean_fn=lambda x: np.float64(0.),
      kernel=kernel12,
      index_points=observation_index_points,
      observation_noise_variance=noise_variance)

## Sample
# gp1_samples = gp1.sample().numpy()
# gp2_samples = gp2.sample().numpy()
# gp12_samples = gp2.sample().numpy()

# ## Plot
# plt.figure(figsize=(14, 7))
# plt.scatter(observation_index_points, gp1_samples, label="GP1")
# plt.scatter(observation_index_points, gp2_samples, label="GP2")
# plt.scatter(observation_index_points, gp12_samples, label="GP12")
# plt.legend()
# plt.show()
# import ipdb; ipdb.set_trace()


## GPR
predictive_index_points = np.random.uniform(-3., 3., (1000, 1))
observation_index_points = np.array([[0.]])
observations = np.array([[0.]])

def mean_fn1(x):
  mean = tf.squeeze(x * 3)
  return mean

def mean_fn2(x):
  mean = tf.squeeze(x * -1)
  return mean

def mean_fn12(x):
  mean = mean_fn1(x) + mean_fn2(x)
  return mean

gprm1 = tfd.GaussianProcessRegressionModel(
    mean_fn=mean_fn1,
    kernel=kernel1,
    index_points=predictive_index_points,
    observation_index_points=observation_index_points,
    observations=observations,
    observation_noise_variance=noise_variance,
    predictive_noise_variance=0.)

gprm2 = tfd.GaussianProcessRegressionModel(
    mean_fn=mean_fn2,
    kernel=kernel2,
    index_points=predictive_index_points,
    observation_index_points=observation_index_points,
    observations=observations,
    observation_noise_variance=noise_variance,
    predictive_noise_variance=0.)

gprm12 = tfd.GaussianProcessRegressionModel(
    mean_fn=mean_fn12,
    kernel=kernel12,
    index_points=predictive_index_points,
    observation_index_points=observation_index_points,
    observations=observations,
    observation_noise_variance=noise_variance,
    predictive_noise_variance=0.)

gpr1_samples = gprm1.sample()
gpr2_samples = gprm2.sample()
gpr12_samples = gprm12.sample()

## Plot
plt.figure(figsize=(14, 8))
# plt.subplot(311)
plt.title(r"$K_1=$Exponentiated quadratic")
plt.scatter(predictive_index_points, gpr1_samples, label="GP1")
# plt.subplot(312)
plt.title(r"$K_2=$Periodic")
plt.scatter(predictive_index_points, gpr2_samples, label="GP2")
# plt.subplot(313)
plt.title(r"$K_1 + K_2$")
plt.scatter(predictive_index_points, gpr12_samples, label="GP12")
plt.legend()
plt.show()

import ipdb; ipdb.set_trace()
