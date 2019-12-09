import numpy as np

from multiclassLRTrain import multiclassLRTrain

def trainModel(x, y, eta, lambda_, iter_):
    param = {}
    param['lambda'] = lambda_     # Regularization term
    param['maxiter'] = iter_     # Number of iterations
    param['eta'] = eta       # Learning rate

    return multiclassLRTrain(x, y, param)
"""
def trainModel(x, y):
    param = {}
    param['lambda'] = 0.01     # Regularization term
    param['maxiter'] = 4000     # Number of iterations
    param['eta'] = 0.09       # Learning rate

    return multiclassLRTrain(x, y, param)
"""