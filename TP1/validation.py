from data_loader import validation_data, X_valid, y_valid
from fonction_decision import prediction
import numpy as np


y_pred = prediction(X_valid)

erreurs = np.array(y_pred != y_valid)
erreurs = erreurs.astype(int)
erreurs = np.sum(erreurs)

if __name__ == "__main__":
    print("nombre d'erreurs :", erreurs)
    print("Pourcentage des erreurs :", 100*erreurs/y_valid.size, "%")
