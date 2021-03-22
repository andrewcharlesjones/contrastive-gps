import sys
sys.path.append("../regression")
from gpr import ContrastiveGaussianProcessRegression
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels


def test_gpr():
	# Generate data
	n = 100
	p = 1
	X = np.random.normal(size=(n, p))
	groups = np.random.binomial(n=1, p=0.5, size=n)
	beta_shared = np.random.normal(size=p)
	beta_fg = np.random.normal(size=p)
	y = X @ beta_shared + (X @ beta_fg) * groups
	
	# Initialize model
	kernel_shared = tfk.Linear()
	kernel_fg = tfk.Linear()
	gpr = ContrastiveGaussianProcessRegression(kernel_shared=kernel_shared, kernel_fg=kernel_fg)

	# Fit
	gpr.fit(X, y, groups)

	# Predict
	preds = gpr.predict(X, groups)

	assert np.allclose(preds, y, atol=1e-1)

	# plt.scatter(X, preds, label="preds")
	# plt.scatter(X, y, label="truth")
	# plt.legend()
	# plt.show()


if __name__ == "__main__":
	test_gpr()
