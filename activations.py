import copy

import numpy as np

# Return the edge strength (e by 1) calculated by a logistic function
# Inputs: edge features (e by w) and edge feature weights (vector w)
'''
sigmoid() activation function
NOTE 1: high accuracy ~0.92
NOTE 2: stable, always above 0.90
'''
def logistic_edge_strength(features, w):
    '''
    args:
        features (np array): (e, w)
        w (np array): (w,)
    return:
        strength (np array): (e, 1)
    '''
    return  1.0 / (1+np.exp(-features.dot(w)))


# Calculate the gradient of edge strength functioin with
# respect to edge feature weights, returns a matrix of gradients (e by w)
# Equation: dStrength/dw = features * edge_strength * (1-edge_strength)
def logistic_strength_gradient(features, edge_strength):
    '''
    args:
        features: (e, w)
        edge_strength: (e, 1)
    return:
        grad: (e, w)
    '''
    logistic_slop = np.multiply(edge_strength, (1-edge_strength))[:,np.newaxis]
    return features.multiply(logistic_slop)


'''
tanh() activation function with ReLu()
NOTE 1: Not working, random walk diverges
NOTE 2: With ReLU(), it works. It seems we have to force the activated value to be positive
NOTE 3: Accuracy: 0.6-0.7 (training) (with 100 iterations, other default)
NOTE 4: Unstable accuracy value varies

TODO: is there anything wrong the the gradient?
'''
def tanh_edge_strength(features, w):
    exp_positive = np.exp(features.dot(w))
    exp_negative = np.exp(-features.dot(w))
    return np.maximum((exp_positive - exp_negative) / (exp_positive + exp_negative), 1e-5)

def tanh_strength_gradient(features, strength):
    relu_strength = copy.deepcopy(strength)
    relu_strength[relu_strength > 0] = 1
    relu_strength[relu_strength <= 0] = 1e-5
    grad = (relu_strength * (1.0 - strength * strength))[:,np.newaxis]
    return features.multiply(grad)

'''
softplus() activation function
NOTE 1: This one seems pretty good, ~0.93 accuracy, max ~0.97
NOTE 2: stable, performance similar to sigmoid
'''
def softplus_edge_strength(features, w):
    return np.log(1 + np.exp(features.dot(w)))

def softplus_strength_gradient(features, strength):
    grad = (strength / (1.0 + strength))[:,np.newaxis]
    return features.multiply(grad)
