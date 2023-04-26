import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt


# load data from text file (note that the initial comments have been deleted from the original file)

data = np.loadtxt("cadata.txt")
# the first column corresponds to the target variables; the remaining ones are the features
y, X = data[:, 0], data[:, 1:]

# YOUR TASKS
# Split data into training, validation, and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

print(X_tr.shape, X_val.shape, X_test.shape)
print(y_tr.shape, y_val.shape, y_test.shape)


# Lambda Function
normalize = lambda z, z_min, z_max: (z - z_min) / (z_max - z_min + 1e-5)

# Normalize X
x_min, x_max = X_tr.min(0, keepdims=True)[0], X_tr.max(0, keepdims=True)[0]
X_train_n, X_test_n, X_val_n, X_test_n_full = normalize(X_tr, x_min, x_max), normalize(X_test, x_min, x_max), normalize(X_val, x_min, x_max), normalize(X_train, x_min, x_max)
# Normalize y
y_min, y_max = y_tr.min(0, keepdims=True)[0], y_tr.max(0, keepdims=True)[0]
y_train_n, y_test_n, y_val_n, y_test_n_full = normalize(y_tr, y_min, y_max), normalize(y_test, y_min, y_max), normalize(y_val, y_min, y_max), normalize(y_train, y_min, y_max)










# - conduct a grid search for the number of nearest neighbors using the validation set;
# - assess the final quality of the model using the root mean squared error (RMSE; check out on your own)
# - generate a plot "predictions vs. true labels"

n_neighbors_range = range(2, 30)
# For each k, we compute the accuracy on the validation set
# (accuracy = percentage of correctly classified instances)
validation_accuracies = []
for k in n_neighbors_range:
    # instantiate model and fit model
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train_n, y_train_n)
    # compute accuracy on validation set
    val_preds = model.predict(X_val_n)
    rmse= np.sqrt(mean_squared_error(val_preds, y_val_n))
    validation_accuracies.append(rmse)

# plot the induced validation accuracies
plt.plot(n_neighbors_range, validation_accuracies)
plt.xlabel('Number of neighbors')
plt.xticks(n_neighbors_range)
plt.ylabel('Validation RMSE')
plt.show()


# select best k
best_k_idx = np.argmin(np.array(validation_accuracies))
best_k = n_neighbors_range[best_k_idx]
print("Best model parameter: {}".format(best_k))
# fit final model
final_model = KNeighborsRegressor(n_neighbors=best_k)
final_model.fit(X_test_n_full, y_test_n_full)
# get predictions for test set
preds = final_model.predict(X_val_n)
final_accuracy = np.sqrt(mean_squared_error(y_val_n, preds))
print("RMSE on test set: {}".format(final_accuracy))


# plot val_preds vs y_val
plt.scatter(y_val_n, preds)
# plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], '--', color='red')
plt.xlabel('True Labels')
plt.ylabel('Predictions')
plt.title('Predictions vs. True Labels on Validation Set')
plt.show()

# - try to improve the performance!
