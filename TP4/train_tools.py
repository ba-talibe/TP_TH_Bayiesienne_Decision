import os
import numpy as np
import pandas as pd
from tools import test, get_error_rate
from os.path import join
from fonction_decision import prediction




def get_optimal_boundary(filename="tp1_data_train.txt", filedir="tp1_data"):

    train_data = pd.read_csv(join(filedir, filename), names=['x', 'y'])

    def get_delta(y):
        
        i0 =(np.min(train_data[train_data['y'] == classes[y[0]] ]["x"]  ), np.max(train_data[train_data['y'] == classes[y[0]]]["x"] )) 
        i1 =(np.min(train_data[train_data['y'] == classes[y[1]] ]["x"]  ), np.max(train_data[train_data['y'] == classes[y[1]]]["x"] ))


        [i0, i1] = sorted([i0, i1], key=lambda x: x[1])

        delta = [np.min([i0[1], i1[0]]), np.max([i0[1], i1[0]])]
        delta = [np.floor(delta[0]), np.ceil(delta[1])]
        return delta

    def reduce_interval(delta, classes):
        i = 0
        while np.abs(delta[0]-delta[1])> 10**-1:
            delta_i = np.arange(delta[0], delta[1], 10**-i)
            if len(delta_i) < 2:
                break

            new_train = train_data[train_data["y"].isin(classes)]
            erreurs = np.array([ get_error_rate(new_train, delta, classes) for delta in delta_i ])

            ind_min = np.argsort(erreurs)

            delta = sorted(delta_i[ind_min[:2]])
            i += 1

        return np.mean(delta)





    classes =train_data.y.unique()
    interval_delta1 = get_delta([0, 1])
    print("interval 1 :", interval_delta1)
    delta1 = reduce_interval(interval_delta1, [0, 1])
    interval_delta2 = get_delta([1, 2])
    delta2 = reduce_interval(interval_delta2, [1, 2])

    print("Intervall de recherche 1: ", delta1)
    print("Intervall de recherche 2: ", delta2)

   
    return [delta1, delta2]
    



def test(filename, delta, filedir="TP2/tp2_data"):

    valid_data = pd.read_csv(os.path.join(filedir, filename), names=['x', 'y'])
    classes = valid_data.y.unique()

    y_pred = prediction(valid_data.x, delta)
    valid_data["y_pred"] = y_pred
    erreurs = np.array(y_pred != valid_data.y).astype(int)
    erreurs = np.sum(erreurs)
    erreurs_percent = 100*erreurs/valid_data.y.size

    matrice_confusion = np.zeros((len(classes), len(classes)))
    matrice_confusion = pd.DataFrame(matrice_confusion)
    matrice_confusion.columns =  np.arange(len(classes))
    matrice_confusion.index = matrice_confusion.columns


    for classe1 in  classes:
        for classe2 in   classes:
            matrice_confusion.loc[classe1, classe2] = valid_data[(valid_data.y_pred == classe1) & (valid_data.y == classe2)].shape[0]
    
    # print("erreurs :", erreurs)
    # print("Pourcentage des erreurs :", erreurs_percent, "%")
    # print("matrices de confusion :\n", matrice_confusion)
    return erreurs_percent, matrice_confusion



