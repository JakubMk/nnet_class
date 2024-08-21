import numpy as np
from Dense import Dense
from Activation import Sigmoid, Tanh, ReLU
from Losses import mse, mse_prime, binary_cross_entropy, binary_cross_entropy_prime
from Network import predict, train, accuracy
from matplotlib import pyplot as plt

def plot_decision_boundary(predict_function, network, X, Y):
    x_min, x_max = X[0, :].min(), X[0, :].max()
    y_min, y_max = X[1, :].min(), X[1, :].max()
    h = 0.1
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    net = np.c_[xx.ravel(), yy.ravel()].T
    # Predict the function value for the whole grid
    Z = predict_function(network, net)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)#cmap='jet')
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0,:], X[1,:], c=Y, cmap='jet')
    plt.show()

        
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
Y = np.array([[0], [1], [1], [0]]).T

network = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]

# train
losses = train(network, mse, mse_prime, X, Y, epochs=10000, learning_rate=0.1)
out = predict(network, X)
print("Test accuracy: ")
accuracy(out, Y)
plt.plot(losses)
plt.show()

#plot_decision_boundary(predict, network, X, Y)

# decision boundary plot
points = []
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        z = predict(network, np.array([[x], [y]]))
        points.append([x, y, z[0,0]])

points = np.array(points)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
plt.show()