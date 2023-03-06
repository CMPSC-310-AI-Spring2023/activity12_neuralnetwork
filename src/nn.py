import numpy as np

# Each row is a training example, each column is a feature  [X1, X2, X3]
# 4 by 3 array
X=np.array(([0,0,1],[0,1,1],[1,0,1],[1,1,1]), dtype=float)
y=np.array(([0],[1],[1],[0]), dtype=float)

# Activation function
def sigmoid(t):
    return 1/(1+np.exp(-t))

# Derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)

# Class definition
class NeuralNetwork:
    def __init__(self, x, y):
        TODO: complete implementation


NN = NeuralNetwork(X, y)
for i in range(1500):  # trains the NN 1,000 times
    if i % 100 == 0:
        print("for iteration # " + str(i) + "\n")
        print("Input : \n" + str(X))
        print("Actual Output: \n" + str(y))
        print("Predicted Output: \n" + str(NN.feedforward()))
        # the sum-of-squares error is simply the sum of the difference between each predicted value and the actual value.
        # The difference is squared so that we measure the absolute value of the difference.
        print("Loss: \n" + str(np.mean(np.square(y - NN.feedforward())))) # mean sum squared loss
        print("\n")

NN.train(X, y)
