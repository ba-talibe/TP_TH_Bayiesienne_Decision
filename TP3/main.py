import numpy as np
from train_tools import get_optimal_boundary
from tools import test, show


    
if __name__ == "__main__":
    train_filename = "TP3/tp1_data/tp1_data_train.txt"
    valid_filename = "TP3/tp1_data/tp1_data_valid.txt"
    show(filename=train_filename, filedir="")
    delta = get_optimal_boundary(train_filename, filedir="")

    erreurs, matrice_confusion = test(valid_filename, delta, filedir="")

    print("erreurs :", erreurs)
    print("matrice de confussion \n")
    print(matrice_confusion)
    