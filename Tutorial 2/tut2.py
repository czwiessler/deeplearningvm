import pandas as pd
import torch


fish_data = pd.read_csv('fish2.csv')
print(fish_data)

## Create Dummy Variables and convert to torch
y = fish_data[['Width']].to_numpy() # Target Variable (Numpy)
fish_data = pd.get_dummies(fish_data[['Species', 'Length1', 'Length2', 'Length3', 'Height', 'Width']], dtype=float).to_numpy()
#print(fish_data[0])

X = fish_data # Input variables (Numpy)

X,y = torch.tensor(X), torch.tensor(y)

generator = torch.Generator().manual_seed(42)
idx = torch.randperm(X.shape[0], generator=generator)
X_train, X_test = X[idx[:127],:], X[idx[127:],:]
y_train, y_test = y[idx[:127],:], y[idx[127:],:]

# Lambda Function
standardize = lambda z, z_mean, z_std: (z - z_mean) / (z_std + 1e-5)

# Standardize X
x_mean, x_std = X_train.mean(0, keepdims=True), X_train.std(0, keepdims=True)
X_train_s, X_test_s = standardize(X_train, x_mean, x_std), standardize(X_test, x_mean, x_std)
# Standardize y
y_mean, y_std = y_train.mean(0, keepdims=True), y_train.std(0, keepdims=True)
y_train_s, y_test_s = standardize(y_train, y_mean, y_std), standardize(y_test, y_mean, y_std)

#print(X_train_s)

# 2 Create k Nearest Neighbor Regressor class

class kNNRegressor:
    # k: Number of k Nearest Neighbors
    # p: L_p distance function
    # Note: output label is simple average of k Nearest Neighbors
    def __init__(self, k, p=2.0):
        self.k = k
        self.p = p
        self.train_x = None
        self.train_y = None

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
        y : Array of shape [n_samples, 1]
        """
        ...

        return self

    def predict(self, X, mean=None, std=None):
        """
        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
        """
        # Calculate distance matrix
        dist = ...
        dist = dist.pow(self.p).mean(2).pow(1. / self.p)

        # Get k nearest neighbors
        _, idx_arg = ...

        # Collect labels
        train_y = ...  # First repeat train labels to match idx_arg
        predictions = ...  # Next, gather train labels
        predictions = ...  # Finally, calculate mean

        # Recompute unstandardized data
        if mean is not None and std is not None:
            predictions = ...
        return predictions


# 3. Create Linear Regressor class
class LinearRegression:
    def __init__(self):
        self.weight = None

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
        y : Array of shape [n_samples, 1]
        """

        
        return self

    def predict(self, X, mean=None, std=None):
        """
        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
        """
        # prepend a column of ones
        ones = ...
        X = ...

        # compute predictions
        predictions = ...

        # Recompute unstandardized data
        if mean is not None and std is not None:
            predictions = ...
        return predictions