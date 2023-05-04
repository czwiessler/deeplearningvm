import pandas as pd
import numpy
fish_data = pd.read_csv('fish2.csv')
fish_data

## Create Dummy Variables and convert to torch
import torch
X = fish_data[['Length1', 'Length2', 'Length3', 'Height', 'Width', 'Species']]
X = torch.tensor(pd.get_dummies(X, dtype=float).to_numpy(), dtype=torch.float)
y = torch.tensor(fish_data[['Weight']].to_numpy(), dtype=torch.float)

## Split into train and test set
generator = torch.Generator().manual_seed(42)
idx = torch.randperm(X.shape[0], generator=generator)
X_train, X_test = X[idx[:127],:], X[idx[127:],:]
y_train, y_test = y[idx[:127],:], y[idx[127:],:]

## Standardize Data
# Lambda Function
standardize = lambda z, z_mean, z_std: (z - z_mean) / (z_std + 1e-5)

# Standardize X
x_mean, x_std = X_train[:,:5].mean(0, keepdims=True), X_train[:,:5].std(0, keepdims=True)
X_train[:,:5], X_test[:,:5] = standardize(X_train[:,:5], x_mean, x_std), standardize(X_test[:,:5], x_mean, x_std)
# Standardize y
y_mean, y_std = y_train.mean(0, keepdims=True), y_train.std(0, keepdims=True)
y_train = standardize(y_train, y_mean, y_std)


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

        # prepend a column of ones
        ones = torch.ones((X.shape[0], 1))
        X = torch.cat((ones, X), axis=1)

        # compute weights
        XtX_pinv = torch.linalg.pinv(X.T @ X)
        X = X.float()
        # Xty = torch.matmul(X.T, y)
        Xty = X.T @ y

        # dot: matrix multiplication
        XtX_pinv = XtX_pinv.float()
        self.weight = torch.matmul(XtX_pinv, Xty)
        return self

    def predict(self, X, mean=None, std=None):
        """
        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
        """
        # prepend a column of ones
        ones = torch.ones((X.shape[0], 1))
        X = torch.cat((ones, X), axis=1)

        # compute predictions
        predictions = numpy.dot(X, self.weight)

        # Recompute unstandardized data
        if mean is not None and std is not None:
            std = std.numpy()
            mean = mean.numpy()
            predictions = predictions * std + mean
        return predictions

lr_model = LinearRegression().fit(X_train, y_train)
y_pred = lr_model.predict(X_test, y_mean, y_std)
mse = numpy.mean((y_pred - y_test.numpy()) ** 2)
print(f'The MSE of the linear regression model is: {mse:10.4f}')



import torch.nn as nn


class LinearRegressionGradientDescent:
    def __init__(self, input_features, lr=0.1, epochs=1000):
        self.weight = nn.Parameter(torch.zeros(input_features + 1, 1))
        self.lr, self.epochs = lr, epochs

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
        y : Array of shape [n_samples, 1]
        """
        for _ in range(self.epochs):
            y_pred = self.forward(X)
            loss = nn.functional.mse_loss(y_pred, y)
            loss.backward()
            with torch.no_grad():
                self.weight -= self.lr * self.weight.grad
            self.weight.grad = None
        return self

    def forward(self, X):
        """
        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
        """
        # prepend a column of ones
        """ones = numpy.ones((X.shape[0], 1))
        X = numpy.concatenate((ones, X), axis=1)"""
        ones = torch.ones(X.size(0), 1)
        X = torch.cat((ones, X), 1)

        # compute predictions
        predictions = torch.matmul(X, self.weight)

        return predictions

    def predict(self, X, mean=None, std=None):
        # Execute forward pass
        predictions = self.forward(X)

        # Recompute unstandardized data
        if mean is not None and std is not None:
            predictions = predictions * std + mean

        return predictions

lrd_model = LinearRegressionGradientDescent(12).fit(X_train, y_train)
y_pred = lrd_model.predict(X_test, y_mean, y_std)
#mse = ((y_pred - y_test)**2).mean()
mse = torch.mean((y_pred - y_test)**2)
# print(lrd_model.weight)
print(f'The MSE of the gradient descent linear regression model is: {mse:10.4f}')