from fonction_decision import prediction
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 

def show(filename, filedir="TP2/tp2_data"):
    
    train_data = pd.read_csv(os.path.join(filedir, filename))
    train_data.columns = ['x', 'y']
   
    classes = train_data.y.unique()
    repartitons = {classe : train_data[train_data.y == classe] for classe in classes}
    
    plt.figure()
    for classe in classes:
        card = repartitons[classe].size
        plt.hist(repartitons[classe].x,bins=15, alpha=.7, label=f"str(classe), {card}")
    plt.legend()
    plt.show()



def test(filename, delta, filedir="TP2/tp2_data"):
    
    valid_data = pd.read_csv(os.path.join(filedir, filename))
    valid_data.columns = ['x', 'y']
    x_valid, y_valid = valid_data.x, valid_data.y
    classes = valid_data.y.unique()

    y_pred = prediction(valid_data.y, delta)
    valid_data["y_pred"] = y_pred
    erreurs = np.array(y_pred != y_valid).astype(int)
    erreurs = np.sum(erreurs)

    matrice_confusion = np.array((len(classes), len(classes)))
    for i in range(len(classes)):
        for j in range(len(classes)):
            matrice_confusion[i, j] = valid_data[(valid_data.y_pred == i) & (valid_data.y == j)].to_numpy().shape[0]
    
    print("erreurs :", erreurs)
    print("matrices de confusion :", matrice_confusion)