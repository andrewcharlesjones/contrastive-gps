import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process.kernels import RBF, Matern
from scipy.stats import multivariate_normal

import matplotlib
font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True



n_list = [1, 2, 5, 10, 50, 100]
plt.figure(figsize=(21, 12))

for jj, n in enumerate(n_list):
	m = n

	X = np.expand_dims(np.linspace(-3, 3, n), 1)
	Y = np.expand_dims(np.linspace(-3, 3, m), 1)


	length_scales = np.linspace(0.1, 3, 30)
	min_eigvals = np.zeros(len(length_scales))
	dets = np.zeros(len(length_scales))
	for ii, length_scale in enumerate(length_scales):
		K = np.zeros((n + m, n + m))
		# Kf = RBF(length_scale=length_scale)
		# Kb = RBF()
		# Kf = Matern(length_scale=length_scale, nu=2.5)
		# Kb = Matern(nu=2.5)
		Kf = lambda x, y: x @ y.T * length_scale
		Kb = lambda x, y: x @ y.T * length_scale
		K[:m, :m] = Kb(Y, Y)
		K[m:, m:] = Kf(X, X)
		K[:m, m:] = np.sqrt(Kb(X, X) * Kf(Y, Y)) / np.sqrt(2)
		K[m:, :m] = np.sqrt(Kb(Y, Y) * Kf(X, X)) / np.sqrt(2)

		assert K.shape == (n+m, n+m)

		min_eigvals[ii] = np.min(np.linalg.eigh(K)[0])
		# dets[ii] = np.linalg.det(K)

	plt.subplot(2, 3, jj + 1)
	plt.scatter(length_scales, min_eigvals)
	plt.title(r"$n={}$".format(n*2))
	# plt.xlabel("FG RBF length scale\n(BG length scale=1)")
	plt.xlabel("FG kernel variance\n(BG variance=1)")
	plt.ylabel("Min. eigenvalue of K")
plt.tight_layout()
plt.savefig("./out/rbf_contrastive.png")
plt.show()
import ipdb; ipdb.set_trace()
