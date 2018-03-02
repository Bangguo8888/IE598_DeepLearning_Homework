import numpy as np
import random
import h5py
import time
###########################################################################
#####plot the images..
"""
def PlotImage(vector_in, time_to_wait):
    array0 = 255.0 * np.reshape(vector_in, (28, 28))
    from matplotlib import pyplot as plt
    plt.ion()
    plt.show(block=False)
    plt.figure(figsize=(2, 2))
    plt.imshow(array0, cmap='Greys_r')
    plt.draw()
    plt.show()
    time.sleep(time_to_wait)
    plt.close('all')


# time (in seconds) between each image showing up on the screen
time_to_wait = 1
# number of images from the dataset that will be plotted
N = 100
# plot the images
for i in range(0, 5):
    PlotImage(x_train[i], time_to_wait)
"""
###############################################################################


# initialize the parameters w and b
def initialize_parameters(X, hidden_units, num_class):
    W1 = np.random.randn(hidden_units, X.shape[0]) * np.sqrt(1. / X.shape[0])
    b1 = np.zeros((hidden_units, 1))
    W2 = np.random.randn(num_class, hidden_units) * np.sqrt(1. / hidden_units)
    b2 = np.zeros((num_class, 1))

    parameters = {}
    parameters['W1'] = W1
    parameters['b1'] = b1
    parameters['W2'] = W2
    parameters['b2'] = b2
    return parameters

# the forward propagation process, it will return the loss value of J and the value of A and Z, for the use of calculating the back propagation
# activation_name can be specified as 'sigmoid','relu','tanh' and 'softmax'
def forward_propagation(X, Y, parameters, activation_name):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = activation_forward(Z1, activation_name)

    Z2 = np.dot(W2, A1) + b2
    A2 = activation_forward(Z2, 'softmax')

    cache = {}
    cache['A1'] = A1
    cache['A2'] = A2
    cache['Z1'] = Z1
    cache['Z2'] = Z2

    cost = loss_func(A2, Y)
    return cache, cost

# activation function for the forward process
def activation_forward(x,func_name):
    if func_name.lower() =='relu':
        y = x
        y[x < 0] = 0
    elif func_name.lower() == 'tanh':
        y = (1.0 - np.exp(2*x))/(1.0 + np.exp(2*x))
    elif func_name.lower() == 'sigmoid':
        y = 1.0/(1.0 + np.exp(-x))
    elif func_name.lower() == 'softmax':
        y = np.exp(x)/np.sum(np.exp(x),axis = 0,keepdims = 1)
    else:
        raise NameError("The function's name must be one of them:'ReLU','Tanh','Sigmoid','Softmax'")
    return y

# calculate the value of loss function
def loss_func(A2,Y):
    m = Y.shape[1]
    J = -1.0/m * np.sum(Y * np.log(A2))
    return J

# the backward propagation process, it will return the gradients for all the parameters W and b
# activation_name can be specified as 'sigmoid','relu','tanh' and 'softmax'
def backward_propagation(X, Y, cache, parameters, activation_name):
    m = Y.shape[1]
    W2 = parameters['W2']
    A1 = cache['A1']
    A2 = cache['A2']
    Z1 = cache['Z1']
    Z2 = cache['Z2']

    dZ2 = A2 - Y
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * activation_backward(A1, activation_name)
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {}
    gradients['dW1'] = dW1
    gradients['dW2'] = dW2
    gradients['db1'] = db1
    gradients['db2'] = db2
    return gradients

# activation function for the backward process
def activation_backward(A1,func_name):
    if func_name.lower() =='relu':
        y = np.zeros(A1.shape)
        y[A1 > 0] = 1
    elif func_name.lower() == 'tanh':
        y = 1 - A1**2
    elif func_name.lower() == 'sigmoid':
        y = A1 * (1 - A1)
    else:
        raise NameError("The function's name must be one of them:'ReLU','Tanh','Sigmoid'")
    return y

# update the parameters
def updata_parameters(parameters,gradients,alpha):
    parameters['W1'] -= alpha * gradients['dW1']
    parameters['b1'] -= alpha * gradients['db1']
    parameters['W2'] -= alpha * gradients['dW2']
    parameters['b2'] -= alpha * gradients['db2']
    return parameters


# train the model
# hidden_units : the number of units in the hidden layers
# iterations : how many times the whole train data set will be used to train the model
# activation_name : the name of the activation function for the hidden layer
# alpha : the learning rate
# batch_size : the batch size, default = 50
def train_model(X, Y, num_class, hidden_units, iterations, activation_name, alpha, batch_size = 50):
    parameters = initialize_parameters(X, hidden_units, num_class)
    sample_size = Y.shape[1]
    for i in range(iterations):
        index = np.arange(sample_size)
        random.shuffle(index)
        for j in range(sample_size // batch_size):
            start = j * batch_size
            end = min((j + 1) * batch_size, sample_size)
            X_batch = X[:, index[start:end]]
            Y_batch = Y[:, index[start:end]]
            cache, cost = forward_propagation(X_batch, Y_batch, parameters, activation_name)
            gradients = backward_propagation(X_batch, Y_batch, cache, parameters, activation_name)
            parameters = updata_parameters(parameters, gradients, alpha)

        if i % 10 == 0 or i == iterations - 1:
            print('Cost value after %-5d iterations: %f' % (i, cost))

    return parameters


# make prediction based on the trained parameters
def prediction(X, y, parameters, activation_name):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = activation_forward(Z1, activation_name)

    Z2 = np.dot(W2, A1) + b2
    A2 = activation_forward(Z2, 'softmax')
    y_hat = A2.argmax(axis=0)
    accuracy = sum(y_hat == y) / len(y)
    return y_hat, accuracy


if __name__ == '__main__':
    # load MNIST data
    MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
    x_train = np.float32(MNIST_data['x_train'][:])
    y_train = np.int32(np.array(MNIST_data['y_train'][:, 0]))
    x_test = np.float32(MNIST_data['x_test'][:])
    y_test = np.int32(np.array(MNIST_data['y_test'][:, 0]))
    MNIST_data.close()

    # organize the data
    X_train = x_train.T
    X_test = x_test.T
    num_class = len(set(y_train))

    # one hot encoding
    Y_train = np.zeros((num_class, len(y_train)))
    for i in range(len(y_train)):
        Y_train[y_train[i]][i] = 1

    iterations = 100
    activation_name = 'relu'
    batch_size = 50

    train_accuracy = {}
    test_accuracy = {}
    # try different kinds of learning rate and hidden units
    for alpha in [0.001, 0.005, 0.01, 0.05, 0.1]:
        train_accuracy[alpha] = {}
        test_accuracy[alpha] = {}
        for hidden_units in [5,10,20,50,100,500]:
            # train the model and get the parameters
            parameters = train_model(X_train, Y_train, num_class, hidden_units, iterations, activation_name, alpha,batch_size)
            # make prediction on the training data set
            train_y_hat, train_accuracy[alpha][hidden_units] = prediction(X_train, y_train, parameters, activation_name)
            # make prediction on the test data set
            test_y_hat, test_accuracy[alpha][hidden_units] = prediction(X_test, y_test, parameters, activation_name)
            print("Learning rate : %f\nhidden units: %d" % (alpha, hidden_units))
            print("Accuracy for training data set: %f" % (train_accuracy[alpha][hidden_units]))
            print("Accuracy for test data set: %f\n" % (test_accuracy[alpha][hidden_units]))



################################################################################################
# Conclusion:
# I used the fixed iteration: 100, activation function: relu and batch size: 50
# then I try several learning rate : 0.001, 0.005, 0.01, 0.05, 0.1
# and try several number of hidden units: 5, 10, 20, 50, 100, 500
#
#
# When the number of hidden units are 5, 10, 20, 50, 100, 500, the accuracy for the train data set and test data set are as follows:
#
#            performance['train']['0.001'] = [0.861000, 0.917550, 0.933117, 0.935400, 0.939783, 0.943600]
#            performance['test']['0.001'] = [0.864300, 0.918000, 0.932500, 0.935300, 0.939400, 0.943200]
#
#            performance['train']['0.005'] = [0.893967, 0.940667, 0.963433, 0.974683, 0.979717, 0.983900]
#            performance['test']['0.005'] = [0.889900, 0.934300, 0.957200, 0.965900, 0.971100, 0.975400]
#
#            performance['train']['0.01'] = [0.895833, 0.946467, 0.967917, 0.986917, 0.990050, 0.994183]
#            performance['test']['0.01'] = [0.891000, 0.939100, 0.957700, 0.973000, 0.976500, 0.979700]
#
#            performance['train']['0.05'] = [0.909850, 0.957483, 0.979667, 0.999783, 0.999983, 1.000000]
#            performance['test']['0.05'] = [0.900700, 0.940700, 0.956500, 0.973200, 0.980400, 0.982200]
#
#            performance['train']['0.1'] = [0.908333, 0.958150, 0.980817, 1.000000, 1.000000, 1.000000]
#            performance['test']['0.1'] = [0.897600, 0.939800, 0.950700, 0.973100, 0.980500, 0.982300]
#
################################################################################################