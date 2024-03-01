from fonction_decision import prediction
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 

def show(filename, filedir="TP1/tp1_data"):
    train_data = pd.read_csv(os.path.join(filedir, filename))
    train_data.columns = ['x', 'y']
   
    classes = train_data.y.unique()
    repartitons = {classe : train_data[train_data.y == classe] for classe in classes}
    
    plt.figure()
    for classe in classes:
        card = repartitons[classe].size
        plt.hist(repartitons[classe].x,bins=20, alpha=.7, label=f"str(classe), {card}")
    plt.legend()
    plt.show()



def test(filename, delta, filedir="TP2/tp2_data"):

    valid_data = pd.read_csv(os.path.join(filedir, filename))
    valid_data.columns = ['x', 'y']

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



