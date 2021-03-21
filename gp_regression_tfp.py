import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tf.enable_v2_behavior()


def sinusoid(x, scale=3, shift=0):
  return np.sin(scale * 2 * np.pi * (x[..., 0]) + shift)

# def generate_1d_data(num_training_points, observation_noise_variance):
#   """Generate noisy sinusoidal observations at a random set of points.

#   Returns:
#      observation_index_points, observations
#   """
#   index_points_ = np.random.uniform(-1., 1., (num_training_points, 1))
#   index_points_ = index_points_.astype(np.float64)
#   # y = f(x) + noise
#   observations_ = (sinusoid(index_points_) +
#                    np.random.normal(loc=0,
#                                     scale=np.sqrt(observation_noise_variance),
#                                     size=(num_training_points)))
#   return index_points_, observations_


SCALE_SHARED = 1
SCALE_FG = 2
SHIFT_FG = 0.5*np.pi
SHIFT_SHARED = 0

def generate_1d_contrastive_data(num_training_points, observation_noise_variance):
  """Generate noisy sinusoidal observations at a random set of points.

  Returns:
     observation_index_points, observations
  """

  ### Background data
  index_points_bg = np.random.uniform(-1., 1., (num_training_points, 1))
  index_points_bg = index_points_bg.astype(np.float64)

  # y = f(x) + noise
  observations_bg = (sinusoid(index_points_bg, scale=SCALE_SHARED) +
                   np.random.normal(loc=0,
                                    scale=np.sqrt(observation_noise_variance),
                                    size=(num_training_points)))


  ### Foreground data
  index_points_fg1 = np.random.uniform(-1., 1., (num_training_points, 1))
  index_points_fg1 = index_points_fg1.astype(np.float64)

  index_points_fg2 = np.random.uniform(-1., 1., (num_training_points, 1))
  index_points_fg2 = index_points_fg2.astype(np.float64)

  observations_fg1 = (sinusoid(index_points_fg1, scale=SCALE_FG, shift=SHIFT_FG) + sinusoid(index_points_fg2, scale=SCALE_SHARED, shift=SHIFT_SHARED) +
                   np.random.normal(loc=0,
                                    scale=np.sqrt(observation_noise_variance),
                                    size=(num_training_points)))

  observations_fg2 = (sinusoid(index_points_fg2, scale=SCALE_FG, shift=SHIFT_FG) + sinusoid(index_points_fg2, scale=SCALE_SHARED, shift=SHIFT_SHARED) +
                   np.random.normal(loc=0,
                                    scale=np.sqrt(observation_noise_variance),
                                    size=(num_training_points)))

  observations_fg = np.concatenate([observations_fg1, observations_fg2])
  index_points_fg = np.concatenate([index_points_fg1, index_points_fg2])

  return index_points_bg, observations_bg, index_points_fg, observations_fg


# Generate training data with a known noise level (we'll later try to recover
# this value from the data).
NUM_TRAINING_POINTS = 200
observation_index_points_bg, observations_bg, observation_index_points_fg, observations_fg = generate_1d_contrastive_data(
    num_training_points=NUM_TRAINING_POINTS,
    observation_noise_variance=.1)

# plt.scatter(observation_index_points_bg, observations_bg, label="Background")
# plt.scatter(observation_index_points_fg, observations_fg, label="Foreground")

# predictive_index_points_ = np.expand_dims(np.linspace(-1.2, 1.2, 200, dtype=np.float64), 1)

# plt.plot(predictive_index_points_, sinusoid(predictive_index_points_, scale=SCALE_SHARED),
#          label='True shared fn')
# plt.plot(predictive_index_points_, sinusoid(predictive_index_points_, scale=SCALE_FG, shift=SHIFT_FG) + sinusoid(predictive_index_points_, scale=SCALE_SHARED, shift=SHIFT_SHARED),
#          label='True fg fn')

# plt.plot(predictive_index_points_, sinusoid(predictive_index_points_, scale=SCALE_FG, shift=SHIFT_FG),
#          label='True fg-specific fn')
# plt.legend()
# plt.show()
# import ipdb; ipdb.set_trace()


# def build_gp(amplitude, length_scale, observation_noise_variance):
#   """Defines the conditional dist. of GP outputs, given kernel parameters."""

#   # Create the covariance kernel, which will be shared between the prior (which we
#   # use for maximum likelihood training) and the posterior (which we use for
#   # posterior predictive sampling)
#   kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)

#   # Create the GP prior distribution, which we will use to train the model
#   # parameters.
#   return tfd.GaussianProcess(
#       kernel=kernel,
#       index_points=observation_index_points_,
#       observation_noise_variance=observation_noise_variance)

# gp_joint_model = tfd.JointDistributionNamed({
#     'amplitude': tfd.LogNormal(loc=0., scale=np.float64(1.)),
#     'length_scale': tfd.LogNormal(loc=0., scale=np.float64(1.)),
#     'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float64(1.)),
#     'observations': build_gp,
# })

def build_bg_gp(amplitude_shared, length_scale_shared, observation_noise_variance):
  """Defines the conditional dist. of GP outputs, given kernel parameters."""

  # Create the covariance kernel, which will be shared between the prior (which we
  # use for maximum likelihood training) and the posterior (which we use for
  # posterior predictive sampling)
  kernel = tfk.ExponentiatedQuadratic(amplitude_shared, length_scale_shared)

  # Create the GP prior distribution, which we will use to train the model
  # parameters.
  return tfd.GaussianProcess(
      kernel=kernel,
      index_points=observation_index_points_bg,
      observation_noise_variance=observation_noise_variance)

def build_fg_gp(amplitude_shared, length_scale_shared, amplitude_fg, length_scale_fg, observation_noise_variance):
  """Defines the conditional dist. of GP outputs, given kernel parameters."""

  # Create the covariance kernel, which will be shared between the prior (which we
  # use for maximum likelihood training) and the posterior (which we use for
  # posterior predictive sampling)
  kernel_shared = tfk.ExponentiatedQuadratic(amplitude_shared, length_scale_shared)
  kernel_fg = tfk.ExponentiatedQuadratic(amplitude_fg, length_scale_fg)
  kernel = kernel_shared + kernel_fg

  # Create the GP prior distribution, which we will use to train the model
  # parameters.
  return tfd.GaussianProcess(
      kernel=kernel,
      index_points=observation_index_points_fg,
      observation_noise_variance=observation_noise_variance)

contrastive_gp_joint_model = tfd.JointDistributionNamed({
    'amplitude_shared': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'amplitude_fg': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'length_scale_shared': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'length_scale_fg': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'observations_bg': build_bg_gp,
    'observations_fg': build_fg_gp
})



x = contrastive_gp_joint_model.sample()
lp = contrastive_gp_joint_model.log_prob(x)

print("sampled {}".format(x))
print("log_prob of sample: {}".format(lp))



# Create the trainable model parameters, which we'll subsequently optimize.
# Note that we constrain them to be strictly positive.

constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

amplitude_shared_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='amplitude_shared',
    dtype=np.float64)

amplitude_fg_var = tfp.util.TransformedVariable(
    initial_value=2.,
    bijector=constrain_positive,
    name='amplitude_fg',
    dtype=np.float64)

length_scale_shared_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='length_scale_shared',
    dtype=np.float64)

length_scale_fg_var = tfp.util.TransformedVariable(
    initial_value=2.,
    bijector=constrain_positive,
    name='length_scale_fg',
    dtype=np.float64)

observation_noise_variance_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='observation_noise_variance_var',
    dtype=np.float64)

trainable_variables = [v.trainable_variables[0] for v in 
                       [amplitude_shared_var,
                       amplitude_fg_var,
                       length_scale_shared_var,
                       length_scale_fg_var,
                       observation_noise_variance_var]]


# Use `tf.function` to trace the loss for more efficient evaluation.
@tf.function(autograph=False, experimental_compile=False)
def target_log_prob(amplitude_shared, length_scale_shared, amplitude_fg, length_scale_fg, observation_noise_variance):
  return contrastive_gp_joint_model.log_prob({
      'amplitude_shared': amplitude_shared,
      'length_scale_shared': length_scale_shared,
      'amplitude_fg': amplitude_fg,
      'length_scale_fg': length_scale_fg,
      'observation_noise_variance': observation_noise_variance,
      'observations_bg': observations_bg,
      'observations_fg': observations_fg
  })



# Now we optimize the model parameters.
num_iters = 1000
optimizer = tf.optimizers.Adam(learning_rate=.01)

# Store the likelihood values during training, so we can plot the progress
lls_ = np.zeros(num_iters, np.float64)
for i in range(num_iters):
  with tf.GradientTape() as tape:
    loss = -target_log_prob(amplitude_shared_var, length_scale_shared_var,
    						amplitude_fg_var, length_scale_fg_var,
                            observation_noise_variance_var)
  grads = tape.gradient(loss, trainable_variables)
  optimizer.apply_gradients(zip(grads, trainable_variables))
  lls_[i] = loss

print('Trained parameters:')
print('amplitude shared: {}'.format(amplitude_shared_var._value().numpy()))
print('length_scale shared: {}'.format(length_scale_shared_var._value().numpy()))
print('amplitude fg: {}'.format(amplitude_fg_var._value().numpy()))
print('length_scale fg: {}'.format(length_scale_fg_var._value().numpy()))
print('observation_noise_variance: {}'.format(observation_noise_variance_var._value().numpy()))


# Plot the loss evolution
# plt.figure(figsize=(12, 4))
# plt.plot(lls_)
# plt.xlabel("Training iteration")
# plt.ylabel("Log marginal likelihood")
# plt.show()



# Having trained the model, we'd like to sample from the posterior conditioned
# on observations. We'd like the samples to be at points other than the training
# inputs.
predictive_index_points_ = np.linspace(-1.2, 1.2, 200, dtype=np.float64)
# Reshape to [200, 1] -- 1 is the dimensionality of the feature space.
predictive_index_points_ = predictive_index_points_[..., np.newaxis]

optimized_kernel_shared = tfk.ExponentiatedQuadratic(amplitude_shared_var, length_scale_shared_var)
optimized_kernel_fg = tfk.ExponentiatedQuadratic(amplitude_fg_var, length_scale_fg_var)
optimized_kernel = optimized_kernel_shared + optimized_kernel_fg


#### Get mean predicted values given one observation
def mean_pred(kernel, x, y, xstar):
  
  return kernel.apply([x], xstar) / kernel.apply([x], [x]) * y
obs_x, obx_y = 0.5, 0.5
preds_combined_kernel = mean_pred(optimized_kernel, obs_x, obx_y, predictive_index_points_)
preds_bg_kernel = mean_pred(optimized_kernel_shared, obs_x, obx_y, predictive_index_points_)
preds_fg_kernel = mean_pred(optimized_kernel_fg, obs_x, obx_y, predictive_index_points_)


plt.figure(figsize=(12, 9))
plt.subplot(211)
plt.title("Data")
plt.scatter(observation_index_points_bg, observations_bg, label="Background", color="orange")
plt.scatter(observation_index_points_fg, observations_fg, label="Foreground", color="blue")
plt.subplot(212)
plt.plot(predictive_index_points_, preds_combined_kernel, label="combined", color="blue")
plt.plot(predictive_index_points_, preds_bg_kernel, label="bg", color="orange")
plt.plot(predictive_index_points_, preds_fg_kernel, label="fg", color="green")
plt.legend()
plt.show()
import ipdb; ipdb.set_trace()

# gprm = tfd.GaussianProcessRegressionModel(
#     kernel=optimized_kernel,
#     index_points=predictive_index_points_,
#     observation_index_points=observation_index_points_,
#     observations=observations_,
#     observation_noise_variance=observation_noise_variance_var,
#     predictive_noise_variance=0.)

gprm_shared = tfd.GaussianProcessRegressionModel(
    kernel=optimized_kernel_shared,
    index_points=predictive_index_points_,
    observation_index_points=observation_index_points_bg,
    observations=observations_bg,
    observation_noise_variance=observation_noise_variance_var,
    predictive_noise_variance=0.)

gprm_fg = tfd.GaussianProcessRegressionModel(
    kernel=optimized_kernel,
    index_points=predictive_index_points_,
    observation_index_points=observation_index_points_fg,
    observations=observations_fg,
    observation_noise_variance=observation_noise_variance_var,
    predictive_noise_variance=0.)

gprm_fg_specific = tfd.GaussianProcessRegressionModel(
    kernel=optimized_kernel_fg,
    index_points=predictive_index_points_,
    observation_index_points=observation_index_points_fg,
    observations=observations_fg,
    observation_noise_variance=observation_noise_variance_var,
    predictive_noise_variance=0.)

# Create op to draw  50 independent samples, each of which is a *joint* draw
# from the posterior at the predictive_index_points_. Since we have 200 input
# locations as defined above, this posterior distribution over corresponding
# function values is a 200-dimensional multivariate Gaussian distribution!
num_samples = 50
samples_shared = gprm_shared.sample(num_samples)
samples_fg = gprm_fg.sample(num_samples)
samples_fg_specific = gprm_fg_specific.sample(num_samples)


# Plot the true function, observations, and posterior samples.
plt.figure(figsize=(12, 9))
# plt.plot(predictive_index_points_, sinusoid(predictive_index_points_),
#          label='True fn')
# plt.scatter(observation_index_points_[:, 0], observations_,
#             label='Observations')
plt.subplot(211)
plt.title("Data")
plt.scatter(observation_index_points_bg, observations_bg, label="Background")
plt.scatter(observation_index_points_fg, observations_fg, label="Foreground")
plt.plot(predictive_index_points_, sinusoid(predictive_index_points_, scale=SCALE_SHARED),
         label='True shared fn')
plt.plot(predictive_index_points_, sinusoid(predictive_index_points_, scale=SCALE_FG, shift=SHIFT_FG) + sinusoid(predictive_index_points_, scale=SCALE_SHARED, shift=SHIFT_SHARED),
         label='True fg fn')
plt.legend()
plt.subplot(212)
plt.title("Posterior samples")
plt.scatter(observation_index_points_bg, observations_bg, label="Background")
plt.scatter(observation_index_points_fg, observations_fg, label="Foreground")
for i in range(num_samples):
  plt.plot(predictive_index_points_, samples_shared[i, :], c='r', alpha=.1,
           label=r'$\mathcal{GP}_s$ posterior samples' if i == 0 else None)
  plt.plot(predictive_index_points_, samples_fg[i, :], c='g', alpha=.1,
           label=r'$\mathcal{GP}_{s+f}$ posterior samples' if i == 0 else None)
  plt.plot(predictive_index_points_, samples_fg_specific[i, :], c='purple', alpha=.1,
           label=r'$\mathcal{GP}_f$ posterior samples' if i == 0 else None)
leg = plt.legend(loc='upper right')
for lh in leg.legendHandles:
    lh.set_alpha(1)
plt.tight_layout()
plt.savefig("./out/sinusoidal_contrastive_gp.png")
plt.show()

import ipdb; ipdb.set_trace()
