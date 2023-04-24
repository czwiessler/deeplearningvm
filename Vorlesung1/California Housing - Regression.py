import numpy

# load data from text file (note that the initial comments have been deleted from the original file)
data = numpy.loadtxt("cadata.txt")

# the first column corresponds to the target variables; the remaining ones are the features
y, X = data[:,0], data[:,1:]

# YOUR TASKS
# - split up the data into training, validation, test sets
# - conduct a grid search for the number of nearest neighbors using the validation set;
# - assess the final quality of the model using the root mean squared error (RMSE; check out on your own)
# - generate a plot "predictions vs. true labels"
# - try to improve the performance!