import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import multivariate_normal as mvn

import matplotlib
font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

class Kernel():

	def __init__():
		pass


class K3(Kernel):

	def __init__(sigma2=1, a=1, b=1):
		self.sigma2 = sigma2
		self.a = a
		self.b = b
		self.equation_string = r"$\frac{\sigma^2}{(a^2 (d-l)^2 + 1)^{p/2}} \exp\{-\frac{b^2 \|x - y\|^2}{a^2 (d-l)^2 + 1}\}$"

	# def 

def K3(x, y, d, l, sigma2=1, a=0.1, b=1):

	p = len(x)
	assert len(x) == len(y)
	scaling_term = sigma2**2 / (a**2 * (d - l)**2 + 1)**(p / 2.0)
	exp_term = -(b**2 * np.sum(((x - y)**2))) / (a**2 * (d - l)**2 + 1)

	return scaling_term * np.exp(exp_term)

if __name__ == "__main__":

	n = 50
	m = 50
	x = np.expand_dims(np.linspace(-3, 3, n), 1)
	y = np.expand_dims(np.linspace(-3, 3, m), 1)
	dx = np.ones(n)
	dy = np.zeros(m)

	X = np.vstack([x, y])
	d = np.concatenate([dx, dy])


	plt.figure(figsize=(10, 10))

	for plot_idx, a in enumerate([0.1, 1, 10]):
		

		K = np.zeros((n+m, n+m))
		for ii in range(n+m):
			for jj in range(ii + 1):
				currK = K3(X[ii], X[jj], d[ii], d[jj], a=a)
				K[ii, jj] = currK
				K[jj, ii] = currK

		# sns.heatmap(K)
		# plt.show()

		sample = mvn.rvs(mean=np.zeros(n+m), cov=K)
		equation_string = r"$k((x, d), (y, l)) = \frac{\sigma^2}{(a^2 (d-l)^2 + 1)^{p/2}} \exp\left\{-\frac{b^2 \|x - y\|^2}{a^2 (d-l)^2 + 1}\right\}$"

		
		plt.subplot(3, 1, plot_idx + 1)
		plt.plot(X[:n], sample[:n], label="FG")
		plt.plot(X[n:], sample[n:], label="BG")
		plt.title(r"$a={}$".format(a))
		plt.legend()
	
	plt.suptitle(equation_string)
	plt.tight_layout()
	plt.savefig("./out/spatiotemporal_kernel.png")
	plt.show()
	import ipdb; ipdb.set_trace()







