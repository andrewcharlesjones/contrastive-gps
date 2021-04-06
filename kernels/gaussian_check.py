import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import seaborn as sns

n = 100
p = 10
X = mvn.rvs(mean=np.zeros(p), cov=2 * np.eye(p), size=n)
Y = mvn.rvs(mean=np.zeros(p), cov=3 * np.eye(p), size=n)

outers = []
for ii in range(n):
	for jj in range(n):
		outer = np.outer(X[ii, :], X[jj, :])
		outers.append(outer)

mean_outer = np.mean(np.array(outers), axis=0)
print(mean_outer)
sns.heatmap(mean_outer)
plt.show()
import ipdb; ipdb.set_trace()