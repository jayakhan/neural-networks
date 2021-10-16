"""Binary Classification using Numpy Operations"""

import numpy as np


def calculate_accuracy(y_hat_class, Y):
    """Calculate accuracy."""
    acc = np.sum(Y == y_hat_class) / len(Y)
    return acc

def sigmoid(x):
    """Sine Function."""
    return 1.0/(1.0 + np.exp(-x))

def predict(bn, X):
    """Predict."""
    # Pull weights and bias from FFNN model
    w_b = [p.detach().numpy() for i, p in bn.net.named_parameters()]

    init_weight_1 = w_b[0]
    init_bias_1 = w_b[1].reshape(-1, 1)
    init_weight_2 = w_b[2]
    init_bias_2 = w_b[3].reshape(-1, 1)
    
    l1 = sigmoid((X @ init_weight_1.T) + init_bias_1.T)
    y_hat = sigmoid((l1 @ init_weight_2.T) + init_bias_2.T)

    return y_hat
