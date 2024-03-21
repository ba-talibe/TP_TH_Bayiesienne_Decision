import numpy as np

def get_vector_predictor(cmin, cmax):
    def predictor(x):
        X = np.hstack([x, np.ones((x.shape[0], 1))])
        a = (cmin[1]-cmax[1])/(cmin[0]-cmax[0])
        b = cmax[1] - a*cmax[0]
        theta = [a, -1, b]
        return np.where(X@theta <= 0, 0, 1)

    return predictor

def get_predictor(cmin, cmax):
    def inner(x):
        a = (cmin[1]-cmax[1])/(cmin[0]-cmax[0])
        b = cmax[1] - a*cmax[0]
        return 0 if x[1] >= x[0]*a + b else 1
    return inner


if __name__ == '__main__':
    prediction1 = get_predictor((-1, 0), (1, 2))
    prediction2= get_vector_predictor((-1, 0), (1, 2))
    print(prediction1(np.array([0, 4])))
    print(prediction2(np.array([[0, 4]])))