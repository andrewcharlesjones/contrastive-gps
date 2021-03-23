import numpy as np


n = 100
p = 10
n_foreground = 50
n_background = n - n_foreground
X = np.random.normal(size=(n, p))
XXT = X @ X.T
print(np.all(np.linalg.eigvals(XXT) > -1e-10))

XXT_contrastive = XXT.copy()
XXT_contrastive[-n_foreground:, -n_foreground:] *= 4
XXT_contrastive[:n_background, -n_foreground:] *= 2.0
XXT_contrastive[-n_foreground:, :n_background] *= 2.0


print(np.all(np.linalg.eigvals(XXT_contrastive) > -1e-10))
print(np.min(np.linalg.eigvals(XXT_contrastive)))
import ipdb

ipdb.set_trace()
