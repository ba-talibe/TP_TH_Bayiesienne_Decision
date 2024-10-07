import numpy as np
import pandas as pd
from os.path import join
from tools import test, get_dm_dM





def get_optimal_boundary(filename="tp1_data_train.txt", filedir="tp1_data"):
    train_data = pd.read_csv(join(filedir, filename), names=['x', 'y'])

    delta = get_dm_dM(train_data)

    print("Intervall de recherche : ", delta)

    i = 0
    while np.abs(delta[0]-delta[1])> 10**-2:
        
        delta_i = np.arange(delta[0], delta[1], 10**-i)
        if len(delta_i) < 2:
            break

        erreurs = np.array([ test(filename,delta, filedir=filedir)[0] for delta in delta_i ])

        ind_min= np.argsort(erreurs)

        delta = sorted(delta_i[ind_min[:2]])
        i+= 1

    return np.mean(delta)
    


