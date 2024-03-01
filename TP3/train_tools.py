import numpy as np
import pandas as pd
from os.path import join
from tools import test




def get_optimal_boundary(filename="tp1_data_train.txt", filedir="tp1_data"):
    train_data = pd.read_csv(join(filedir, filename))

    train_data.columns = ['x', 'y']

    classes =train_data.y.unique()
    i0 =(np.min(train_data[train_data['y'] == classes[0] ]["x"]  ), np.max(train_data[train_data['y'] == classes[0]]["x"] )) 
    i1 =(np.min(train_data[train_data['y'] == classes[1] ]["x"]  ), np.max(train_data[train_data['y'] == classes[1]]["x"] ))

    [i0, i1] = sorted([i0, i1], key=lambda x: x[1])

    delta = [np.min([i0[1], i1[0]]), np.max([i0[1], i1[0]])]
    delta = [np.floor(delta[0]), np.ceil(delta[1])]
    print("Intervall de recherche : ", delta)

    while np.abs(delta[0]-delta[1])> 10**-2:
        
        delta_i = np.arange(delta[0], delta[1], 10**-2)
        if len(delta_i) < 2:
            break

        erreurs = np.array([ test(filename,delta, filedir=filedir)[0] for delta in delta_i ])

        ind_min= np.argsort(erreurs)

        delta = sorted(delta_i[ind_min[:2]])

    return np.mean(delta)
    


