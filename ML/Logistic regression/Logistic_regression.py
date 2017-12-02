import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    '''1 / (1 + e^(-x))'''
    return 1 / (1 + np.exp(-x))

def log_likelihood(features, target, weights):
    '''
        U = sum(target * weights_tr * features - log(1 + exp(weights_tr * features)))
    '''
    scores = np.dot(features, weights)
    
    scores = np.dot(features, weights)
    ll = np.sum( target * scores - np.log(1 + np.exp(scores)) )
    return ll

def grad(features, target, predictions):
    '''
        grad(U) = features_tr * (target - predictions)
    '''
    
    diff = target - predictions
    return np.dot(features.T, diff)

def logistic_regression(features, target, num_steps, learning_rate):
    # initialize weights
    features = np.hstack((np.ones((features.shape[0], 1)),features))
    weights = np.zeros(features.shape[1])
    
    # iterative process
    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with gradient
        gradient = grad(features, target, predictions)
        weights += learning_rate * gradient
        
        # Print log-likelihood every so often
        if step % 10000 == 0:
            print(log_likelihood(features, target, weights))
        
    return weights

def run():
    np.random.seed(12)
    num_observations = 5000

    x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

    simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
    simulated_labels = np.hstack((np.zeros(num_observations),
                              np.ones(num_observations)))
    
    weights = logistic_regression(simulated_separableish_features, simulated_labels,
                     num_steps = 100000, learning_rate = 5e-5)
    
if __name__ == '__main__':
    run()
