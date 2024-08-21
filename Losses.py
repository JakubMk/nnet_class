import numpy as np

def mse(Y_true, Y_pred):
    # it can be replaced just by one np.mean()
    return np.mean(np.mean(np.power(Y_pred - Y_true, 2), axis=0, keepdims=True))

def mse_prime(Y_true, Y_pred):
    n, _ = Y_true.shape
    return 2/n * (Y_pred - Y_true) #/ np.size(Y_true)

def binary_cross_entropy(Y_true, Y_pred):
    return -1 * np.mean(np.add(np.multiply(Y_true,np.log(Y_pred)) , np.multiply((1 - Y_true),np.log(1 - Y_pred))))

def binary_cross_entropy_prime(Y_true, Y_pred):
    return np.subtract(np.divide((1 - Y_true) , (1 - Y_pred)) , np.divide(Y_true , Y_pred)) #/ np.size(Y_true)