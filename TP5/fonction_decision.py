import numpy as np

def classify(cmin, cmax):
    def inner(x):
        a = (cmin[1]-cmax[1])/(cmin[0]-cmax[0])
        b = cmax[1] - a*cmax[0]
        return 
    return inner


def prediction(x,  cmin, cmax):
    classifer = classify(cmin, cmax)
    return np.vectorize(classifer)(x)