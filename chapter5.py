#%%
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import make_moons


# %%
# Import iris dataset, scale it and train a svm with a small C
iris = datasets.load_iris()
X = iris['data'][:, (2, 3)] # petal length and width
y = (iris['target'] == 2).astype(np.float64) #1s on Iris Virginica and 0s on rest

svm_clf = Pipeline((
  ('scaler', StandardScaler()),
  ('linear_svc', LinearSVC(C=1, loss='hinge'))
))

svm_clf.fit(X, y)
svm_clf.predict([[5.5, 1.7]])

# %%
# Add Polynomial features to pipeline
polynomial_svm_clf = Pipeline((
  ('poly_features', PolynomialFeatures(degree=2)),
  ('scaler', StandardScaler()),
  ('linear_svc', LinearSVC(C=10, loss='hinge'))
))
polynomial_svm_clf.fit(X, y)

# %%
# Use polynomial kernel instead (more performant)
poly_kernel_svm_clf = Pipeline((
  ('scaler', StandardScaler()),
  ('svm_clf', SVC(kernel='poly', degree=3, coef0=1, C=5))
))
poly_kernel_svm_clf.fit(X, y)

# %%
# Use Gaussian RBF kernel
rbf_kernel_svm_clf = Pipeline((
  ('scaler', StandardScaler()),
  ('svm_clf', SVC(kernel='rbf', gamma='5', C=0.001))
))
rbf_kernel_svm_clf.fit(X, y)