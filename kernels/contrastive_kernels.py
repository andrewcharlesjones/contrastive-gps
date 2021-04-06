import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import multivariate_normal

import matplotlib
font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


m = 100
n = 100

X = np.expand_dims(np.linspace(-3, 3, n), 1)
Y = np.expand_dims(np.linspace(-3, 3, m), 1)

gamma_list = [1, 1.25, 1.5, 2]

plt.figure(figsize=(15, 10))
for ii, gamma in enumerate(gamma_list):
	K = np.zeros((n + m, n + m))
	Kf = RBF(length_scale=.99)
	Kb = RBF()
	K[:m, :m] = Kb(Y, Y)
	K[m:, m:] = Kf(X, X)
	K[:m, m:] = np.sqrt(Kb(Y, X) * Kf(Y, X)) / gamma
	K[m:, :m] = np.sqrt(Kb(X, Y) * Kf(X, Y)) / gamma

	print(np.min(np.linalg.eigvals(K)))
	import ipdb; ipdb.set_trace()
	K_full = multivariate_normal(mean=np.zeros(m+n), cov=K + 1e-6 * np.eye(n+m))

	S = 100
	samples = K_full.rvs()

	plt.subplot(len(gamma_list), 1, ii+1)
	plt.plot(Y, samples[:m], label="Background", linewidth=3)
	plt.plot(X, samples[m:], label="Foreground", linewidth=3)
	plt.title(r"$\gamma={}$".format(gamma))
	plt.legend()
plt.tight_layout()
plt.savefig("./out/rbf_contrastive.png")
plt.show()
import ipdb; ipdb.set_trace()
