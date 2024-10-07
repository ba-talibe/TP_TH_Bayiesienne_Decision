import numpy as np
from data_loader import train_data,  X_train, y_train

delta = 144


def prediction(x):
    if type(x) in [int, float]:
        return 0 if x < delta else 1
    else:
        result = np.array(x > delta)
        result =result.astype(int)
        return result
