import numpy as np



def prediction(x, delta):
    if type(x) in [int, float]:
        return 0 if x < delta else 1
    else:
        result = np.array(x > delta)
        result = result.astype(int)
        return result
