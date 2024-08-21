import numpy as np

def predict(network, X,):
    output = X
    for layer in network:
        output = layer.forward(output)
    return output

def accuracy(y_pred,y_true):
    acc = np.sum(np.logical_not(np.logical_xor((y_pred>0.5),y_true))) / np.size(y_true)*100
    print(f"Accuracy = {round(acc,2)} %")

def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
    losses = []
    for e in range(epochs):
        error = 0
        # forward
        output = predict(network, x_train)
        # error
        error = loss(y_train, output)
        #losses.append(error)
        
        # backward
        grad = loss_prime(y_train, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)
            
        if verbose:
            if e%100 == 0:
                acc = np.sum(np.logical_not(np.logical_xor((output>0.5),y_train))) / np.size(y_train)*100
                print(f"{e}/{epochs}, error = {error}, accuracy = {round(acc,2)} %")
                losses.append(error)
    return losses
