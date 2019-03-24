import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def initialize_parameter(layer_dims):
    np.random.seed(0)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


#### forward propagation
def linear_forward(A, W, b):

    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache


def relu(Z):

    A = np.maximum(0, Z)

    return A, Z


def softmax(Z):

    e_Z = np.exp(Z)
    A = e_Z / np.sum(e_Z, axis=0)

    return A, Z


def linear_activation_forward(A_prev, W, b, activation):

    if activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    if activation == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)

    cache = linear_cache, activation_cache

    return A,  cache


def L_model_foward(X, parameters):

    L = len(parameters)//2
    A = X
    caches = []
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation="relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation="softmax")
    caches.append(cache)

    return AL, caches


#### compute loss function
def compute_cost(AL, y):

    m = AL.shape[1]
    cost = 1.0/m * np.sum((np.multiply(np.log(AL),y)))
    return cost


#### Back propagation
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1.0/m * np.dot(dZ, A_prev.T)
    db = 1.0/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    return dZ


def softmax_backward(dA, cache):
    Z = cache
    s = np.exp(Z)/np.sum(np.exp(Z), axis=0)
    m = dA.shape[1]
    dZ = (dA)

    return dZ


def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    if activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, y, caches):

    grads = {}
    L = len(caches)
    y = y.reshape(AL.shape)
    dAL = AL - y
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "softmax")

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+2)], current_cache, activation="relu")
        grads["dA" + str(l+1)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp

    return grads


def update_parameter(parameters, grads, learning_rate):

    L = len(parameters)//2
    for l in range(L):
       parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
       parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]

    return parameters


def L_layer_model(X, y, layer_dims, learning_rate=0.0075, num_iterations = 3000):

    costs = []
    parameters = initialize_parameter(layer_dims)

    for i in range(num_iterations):
        AL, caches = L_model_foward(X, parameters)
        # print(np.reshape([int(AL[0,j] >= 0.5) for j in range(len(AL))], AL.shape))
        # AL = np.reshape([int(AL[0,j] >= 0.5) for j in range(len(AL))], AL.shape)
        cost = compute_cost(AL, y)
        grads = L_model_backward(AL, y, caches)
        parameters = update_parameter(parameters, grads, learning_rate)

        # if i%2 == 0:
        #     print("Cost after iterarion %i: %f" %(i, cost))
        #     costs.append(-cost)
    yhat = np.array([int(AL.T[i,j] == np.max(AL.T[i])) for i in range(AL.T.shape[0]) for j in range(AL.T.shape[1])])
    yhat = yhat.reshape(AL.T.shape).T
    accuracy = np.count_nonzero( (yhat+y) == 2)*1.0/y.shape[1]
    # print("Accuracy of algorithm: ", accuracy)

    # plt.figure()
    # plt.plot(np.squeeze(costs))
    # plt.ylabel("Cost")
    # plt.xlabel("Iteration per 100")
    # plt.title("learning rate = " +  str(learning_rate))
    # plt.show()

    # plt.figure()
    # plt.plot(X[0,y[0,:] == 1], X[1, y[0, :] == 1], 'g*', markersize=3, label="class 1")
    # plt.plot(X[0,y[1,:] == 1], X[1, y[1, :] == 1], 'r*', markersize=3, label="class 2")
    # plt.plot(X[0,y[2,:] == 1], X[1, y[2, :] == 1], 'b*', markersize=3, label="class 3")
    # plt.plot(X[0,y[3,:] == 1], X[1, y[3, :] == 1], 'm*', markersize=3, label="class 4")
    # plt.plot(X[0,y[4,:] == 1], X[1, y[4, :] == 1], 'k*', markersize=3, label="class 5")
    # plt.xlabel("X1")
    # plt.ylabel("X2")
    # plt.title("True label of training test - neural network")
    # plt.legend(loc=4)
    #
    #
    # plt.figure()
    # plt.plot(X[0,yhat[0,:] == 1], X[1, yhat[0, :] == 1], 'g*', markersize=3, label="class 1")
    # plt.plot(X[0,yhat[1,:] == 1], X[1, yhat[1, :] == 1], 'r*', markersize=3, label="class 2")
    # plt.plot(X[0,yhat[2,:] == 1], X[1, yhat[2, :] == 1], 'b*', markersize=3, label="class 3")
    # plt.plot(X[0,yhat[3,:] == 1], X[1, yhat[3, :] == 1], 'm*', markersize=3, label="class 4")
    # plt.plot(X[0,yhat[4,:] == 1], X[1, yhat[4, :] == 1], 'k*', markersize=3, label="class 5")
    # plt.xlabel("X1")
    # plt.ylabel("X2")
    # plt.title("Predict label of training test - neural network")
    # plt.legend(loc=4)
    # plt.show()

    return parameters, accuracy


def predict(X_test, y_test, parameters):

    AL, cache = L_model_foward(X_test, parameters)
    yhat = np.array([int(AL.T[i,j] == np.max(AL.T[i])) for i in range(AL.T.shape[0]) for j in range(AL.T.shape[1])])
    yhat = yhat.reshape(AL.T.shape).T
    accuracy = np.count_nonzero( (yhat+y_test) == 2)*1.0/y_test.shape[1]
    print("Accuracy of test algorithm: ", accuracy)

    return accuracy

def predict_realtime(X_test, parameters):
    AL, cache = L_model_foward(X_test, parameters)
    yhat = np.array([int(AL.T[i, j] == np.max(AL.T[i])) for i in range(AL.T.shape[0]) for j in range(AL.T.shape[1])])
    yhat = yhat.reshape(AL.T.shape).T
    if yhat[1,0] == 1:
        return 1
    else:
        return 0


lb = preprocessing.LabelBinarizer()
input = pd.read_csv('./outputCSV/feature_icmp.csv')
# input = pd.read_csv('input.txt')
input = shuffle(input)
X_train = input.iloc[: ,:-1]
y_train = input.iloc[:, -1]
# X_train, X_test, y_train, y_test = train_test_split(X, y)

X_train = X_train.T.as_matrix()
# X_test = X_test.T.as_matrix()
y_train = y_train.T.as_matrix()
y_train = lb.fit_transform(y_train).T
# y_test = y_test.T.as_matrix()
# y_test = lb.transform(y_test).T

y_tr = np.zeros((2,y_train.shape[1]))
# y_te = np.zeros((2,y_test.shape[1]))
for i in range(y_tr.shape[1]):
    if y_train[0,i] == 0:
        y_tr[0,i] = 1
    else:
        y_tr[1,i] = 1

# for i in range(y_te.shape[1]):
#     if y_test[0,i] == 0:
#         y_te[0,i] = 1
#     else:
#         y_te[1,i] = 1


layer_dims = [5,10,10,10,10,10,2]
neural, accuracy = L_layer_model(X_train, y_tr, layer_dims)
print "The process of training is Done"
# test_score = predict(X_test, y_te, neural)

# test = predict_realtime([[0.559665775739],[0.466884879383],[0.481895493381],[0.467945189644],[0.489056239101]], neural)
# print test
# layer_dims = [2,4,5]

# learning_rate = np.arange(0.00005,0.004, 0.00002)
# accuracy_test = []
# accuracy_train = []
# for i in learning_rate:
#     print(i)
#     neural, accuracy = L_layer_model(X_train, y_tr, layer_dims,learning_rate=i)
#     accuracy_train.append(accuracy)
#     test_score = predict(X_test, y_te, neural)
#     accuracy_test.append(test_score)
#
# plt.figure()
# plt.plot(learning_rate, accuracy_train, label='Accuracy in training data')
# plt.plot(learning_rate, accuracy_test, label='Accuracy in Test data')
# plt.xlabel('Learning rate')
# plt.ylabel('Accuracy')
# plt.title('Accuracy of algorithm Deep Neural Network in training and test data')
# plt.legend()
# plt.show()