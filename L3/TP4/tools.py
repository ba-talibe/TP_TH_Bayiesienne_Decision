import os 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fonction_decision import prediction



def show(filename, filedir="TP2/tp2_data", save=True, title=None, delta =None):
    
    train_data = pd.read_csv(os.path.join(filedir, filename), names=['x', 'y'])
   
    classes = train_data.y.unique()
    repartitons = {classe : train_data[train_data.y == classe] for classe in classes}
    
    plt.figure(figsize=(10, 8))
    
    for classe in classes:
        card = repartitons[classe].size
        plt.hist(repartitons[classe].x,bins=20, alpha=.7, label=f"{str(classe)}, {card}")

    plt.title(filename)
    
    if title is not None:
        plt.title(title)
    if delta is not None:
        for deltai in delta:
            plt.axvline(deltai )
            plt.text(deltai, 2, f"{np.round(deltai)}")


    plt.legend()
    plt.show()



def validate(filename, delta, filedir="TP2/tp2_data"):
    
    valid_data = pd.read_csv(os.path.join(filedir, filename), names=['x', 'y'])
    x_valid, y_valid = valid_data.x, valid_data.y
    classes = valid_data.y.unique()

    y_pred = prediction(valid_data.x, delta)
    valid_data["y_pred"] = y_pred
    erreurs = np.array(y_pred != y_valid).astype(int)
    erreurs = np.sum(erreurs)
    erreurs_percent = 100*erreurs/y_valid.size

    matrice_confusion = np.zeros((len(classes), len(classes)))
    matrice_confusion = pd.DataFrame(matrice_confusion)
    matrice_confusion.columns =  np.arange(len(classes))
    matrice_confusion.index = matrice_confusion.columns


    for classe1 in  classes:
        for classe2 in   classes:
            matrice_confusion.loc[classe1, classe2] = valid_data[(valid_data.y_pred == classe1) & (valid_data.y == classe2)].shape[0]
    

    
    if len(classes) == 2:
        plt.figure()

        plt.text(.5, .8, f"erreurs : {erreurs}", fontsize=14, ha='center', va='center')
        plt.text(.5, .7, f"Pourcentage des erreurs : {erreurs_percent} %", fontsize=14, ha='center', va='center')
        plt.text(.5, .5, f"matrices de confusion :\n {matrice_confusion}", fontsize=14, ha='center', va='center')
   
        plt.axis('off')
        plt.title(filename)
        plt.show()

        plt.figure()
        
        plt.text(0, 0, f"vrai positive : {matrice_confusion.loc[0, 0]}", ha='center', fontsize=12, va='center', color='red')
        plt.text(1, 0, f"faux positive : {matrice_confusion.loc[0, 1]}", ha='center', fontsize=12,  va='center', color='red')
        plt.text(0, 1, f"faux negative : {matrice_confusion.loc[1, 0]}", ha='center', fontsize=12, va='center', color='red')
        plt.text(1, 1, f"vrai negative : {matrice_confusion.loc[1, 1]}", ha='center', fontsize=12, va='center', color='red')

        plt.axis("off")
        plt.imshow(matrice_confusion, cmap='YlGnBu', interpolation='nearest')
        plt.title(filename)
        plt.show()
    else:
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrice_confusion, annot=True, cmap='coolwarm', fmt='.1f', linewidths=.5)
        plt.title('Heatmap')
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        plt.show()
        
    
       
    print("erreurs :", erreurs)
    print(f"Pourcentage des erreurs : {erreurs_percent} %")
    print(f"matrices de confusion :\n {matrice_confusion}")
    return erreurs_percent, matrice_confusion



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


def prediction_wrapper(delta, classe):
    def prediction(x):
        if x < delta :
            return classe[0]
        return classe[1]

    return prediction
def get_error_rate(df, delta, classes):
    prediction = prediction_wrapper(delta, classes)

    pred = np.where(df.x <= delta, classes[0], classes[1])
    erreurs = np.array(pred != df.y).astype(int)
    erreurs = np.sum(erreurs)
    return 100*erreurs/df.y.size
