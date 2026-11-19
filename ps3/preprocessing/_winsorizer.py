import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# TODO: Write a simple Winsorizer transformer which takes a lower and upper quantile and cuts the
# data accordingly
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile, upper_quantile):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        self.lower_quantile_ = np.quantile(X, self.lower_quantile)
        self.upper_quantile_ = np.quantile(X, self.upper_quantile)
        return self

    def transform(self, X):
        return np.clip(X, self.lower_quantile_, self.upper_quantile_)

# some test data

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])

w = Winsorizer(lower_quantile=0.2, upper_quantile=0.8)

w.fit(X)

print("Lower cutoff:", w.lower_quantile_)
print("Upper cutoff:", w.upper_quantile_)

X_new = w.transform(X)

print("Original:", X)
print("Winsorised:", X_new)
