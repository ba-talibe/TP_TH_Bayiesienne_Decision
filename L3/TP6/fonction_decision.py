import numpy as np

def get_vector_predictor(mu0, sigma0, mu1, sigma1):
    def predictor(x):
        d0 = np.sum((x-mu0)*(x-mu0), axis=1)
        d1 = np.sum((x-mu1)*(x-mu1), axis=1)
        pred = d0 >= d1
        return pred.astype(int)
    return predictor

def get_predictor(mu0, sigma0, mu1, sigma1):
    def predictor(x):
        d0 = (x-mu0).T@(x-mu0)
        d1 = (x-mu1).T@(x-mu1)
        pred = d0 >= d1
        return pred.astype(int)
    return predictor



if __name__ == '__main__':
    x = np.array([4, 9])
    mu0 = np.array([1, 1])
    mu1 = np.array([.5, .5])
    predictor = get_predictor(mu0, 9, mu1, 10)
    print(predictor(x))
    # print((x-mu0))
    # print((x-mu0)*(x-mu0))
    # print(np.sum((x-mu0)*(x-mu0), axis=1))


    
    
