import numpy as np

def classify(delta):
    def inner(x):
        if x <= delta[0]:
            return 0
        elif x <= delta[1]:
            return 1
        return 2
    return inner


def prediction(x, delta : list):
    classifer = classify(delta)
    return np.vectorize(classifer)(x)