import numpy as np
import pandas as pd

def train_perceptron(train_data, lr=10**-1, iters = 10**4):
    X = train_data[["x1", "x2"]].to_numpy()
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    y = train_data.y.to_numpy()

    theta = np.array([0.0, 0.0, 0.0])
    activation = lambda x: np.where(x <= 0, 0, 1)

    for _ in range(iters):
        for idx, xi in enumerate(X):
            y_predict = activation(xi@theta)
            correction = lr*(y[idx]-y_predict)
            theta +=  correction*xi
    
    
    x1_min =  train_data["x1"].min() - 10
    x1_max = train_data["x1"].max() + 10
    x2_min = -(theta[0]*x1_min + theta[2])/theta[1]
    x2_max = -(theta[0]*x1_max + theta[2])/theta[1]
    return [x1_min, x2_min], [x1_max, x2_max]
            
def moindre_carrés(train_data, lr=10**-1, iters = 10**4):
    X = train_data[["x1", "x2"]].to_numpy()
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    y = train_data.y.to_numpy()

    theta = np.array([0.0, 0.0, 0.0])
    
            

df = pd.DataFrame()

df.to_numpy()