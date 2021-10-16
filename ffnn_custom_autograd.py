"""Feed Forward Neural Network from scratch using numpy and custom autograd function"""

from gen_data import gen_xor
from sklearn.model_selection import train_test_split
import numpy as np
import ffnn1 as ffnn
import copy
import matplotlib.pyplot as plt


def init_layers(nn_architecture, seed = 99):
    np.random.seed(seed)
    number_of_layers = len(nn_architecture)
    params_values = {}

    bn = ffnn.BinaryLinear(2)
    init_weights = copy.deepcopy(bn.net[0].weight.data).numpy()
    init_bias = copy.deepcopy(bn.net[0].bias.data).numpy()
    init_weights_2 = copy.deepcopy(bn.net[2].weight.data).numpy()
    init_bias_2 = copy.deepcopy(bn.net[2].bias.data).numpy()
    
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        # layer_input_size = layer["input_dim"]
        # layer_output_size = layer["output_dim"]
        if idx == 0:
            params_values['W' + str(layer_idx)] = init_weights
            params_values['b' + str(layer_idx)] = init_bias
        else:
            params_values['W' + str(layer_idx)] = init_weights_2
            params_values['b' + str(layer_idx)] = init_bias_2
    return params_values

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)


def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="sigmoid"):
    print("single forward")

    Z_curr = np.dot(A_prev, W_curr.T) + b_curr
    
    if activation == "sigmoid":
        activation_func = sigmoid
    else:
        raise Exception('Non-supported activation function')
        
    return activation_func(Z_curr), Z_curr


def full_forward_propagation(X, params_values, nn_architecture):
    print('full_forward')
    memory = {}
    A_curr = X
    

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        A_prev = A_curr
        activ_function_curr = layer["activation"]
        
        W_curr = params_values["W" + str(layer_idx)]
        b_curr = params_values["b" + str(layer_idx)]
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)
        
        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr
       
    return A_curr, memory


def get_cost_value(Y_hat, Y):
    m = Y_hat.shape[1]
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)


def get_accuracy_value(Y_hat, Y):
    r = np.sum(Y == Y_hat) / len(Y)
    print(r)
    return r
    # Y_hat_ = convert_prob_into_class(Y_hat)
    # return (Y_hat_ == Y).all(axis=0).mean()

def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="sigmoid"):
    m = A_prev.shape[1]
    
    if activation == "sigmoid":
        backward_activation_func = sigmoid_backward
    else:
        raise Exception('Non-supported activation function')
    
    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    dW_curr = np.dot(dZ_curr, Z_curr.T) / m
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    dA_prev = np.dot(dZ_curr, W_curr)

    return dA_prev, dW_curr, db_curr

def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}
    m = Y.shape[1]
    Y = Y.reshape(Y_hat.shape)
   
    dA_prev =- (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
    
    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        activ_function_curr = layer["activation"]
        
        dA_curr = dA_prev
        
        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]
        
        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
        
        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr
    
    return grads_values


def update(params_values, grads_values, nn_architecture, learning_rate):
    for layer_idx, layer in enumerate(nn_architecture):
        params_values["W" + str(layer_idx+1)] -= learning_rate * grads_values["dW" + str(layer_idx+1)]        
        params_values["b" + str(layer_idx+1)] -= learning_rate * grads_values["db" + str(layer_idx+1)]

    return params_values

def train(X, Y, nn_architecture, epochs, learning_rate):
    params_values = init_layers(nn_architecture, 2)
    cost_history = []
    accuracy_history = []
    
    for i in range(epochs):
        print('train')
        Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)

        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)
        y_hat_class = np.where(Y_hat < 0.5, 0, 1)

        accuracy = get_accuracy_value(y_hat_class, Y)
        accuracy_history.append(accuracy)
        
        grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
        #params_values = update(params_values, grads_values, nn_architecture, learning_rate)
    return cost_history, accuracy_history

def plot_training_progress(loss, accuracy):
        """Plot training progress."""
        loss = np.concatenate(loss).ravel().tolist()
        # plt.plot(accuracy)
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        ax[0].plot(loss)
        ax[0].set_ylabel('Loss')
        ax[0].set_title('Training Loss')

        ax[1].plot(accuracy)
        ax[1].set_ylabel('Classification Accuracy')
        ax[1].set_title('Training Accuracy')

        plt.tight_layout()
        plt.show()

def main():
    nn_architecture = [
    {"input_dim": 2, "output_dim": 3, "activation": "sigmoid"},
    {"input_dim": 3, "output_dim": 1, "activation": "sigmoid"},
    ]
    X, Y = gen_xor(400)

    # Split into test and training data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=0.25,
    )
    loss, accuracy = train(X_train, Y_train.reshape(300, 1), nn_architecture, 1000, learning_rate=1e-2)
    plot_training_progress(loss, accuracy)



if __name__ == "__main__":
    main()