import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import multivariate_normal as mvn

def K_linear_contrastive(x1, x2, d1, d2, sigma2_s=1, sigma2_f=1):
	return np.expand_dims((sigma2_s + np.sqrt(sigma2_s*sigma2_f)*d1 + np.sqrt(sigma2_s*sigma2_f)*d2 + sigma2_f*d1*d2), 1) * x1.T @ x2


n = 100
m = 100
p = 1
x1 = np.expand_dims(np.linspace(-2, 2, m), 0)
x2 = np.expand_dims(np.linspace(-2, 2, n), 0)
d1 = np.zeros(m)
d2 = np.ones(n)
sigma2_s = 1.0
sigma2_f = 2.0
group = np.concatenate([d1, d2])
group = ["FG" if d == 1 else "BG" for d in group]
X = np.concatenate([x1.squeeze(), x2.squeeze()])

plt.figure(figsize=(10, 10))
sigma2_f_list = [0.1, 0.5, 1, 2]
n_draws = 100
for ii, sigma2_f in enumerate(sigma2_f_list):
	plt.subplot(2, 2, ii + 1)
	plt.title(r"$\sigma^2_f = {}$".format(sigma2_f))
	plt.ylim([-15, 15])
	for _ in range(n_draws):
		upper_left = K_linear_contrastive(x1, x1, d1, d1, sigma2_s=sigma2_s, sigma2_f=sigma2_f)
		upper_right = K_linear_contrastive(x1, x2, d1, d2, sigma2_s=sigma2_s, sigma2_f=sigma2_f)
		lower_right = K_linear_contrastive(x2, x2, d2, d2, sigma2_s=sigma2_s, sigma2_f=sigma2_f)
		K = np.vstack([
			np.hstack([upper_left, upper_right]),
			np.hstack([upper_right.T, lower_right])])

		assert K.shape == (n+m, n+m)

		
		y = mvn.rvs(mean=np.zeros(m+n), cov=K)

		plt.plot(X[:m], y[:m], alpha=0.2, linewidth=2, color="red")
		plt.plot(X[m:], y[m:], alpha=0.2, linewidth=2, color="blue")

		
plt.show()

# data_df = pd.DataFrame({"X": X, "Y": y, "group": group})
# sns.scatterplot(data=data_df, x="X", y="Y", hue="group")
# plt.show()

import ipdb; ipdb.set_trace()