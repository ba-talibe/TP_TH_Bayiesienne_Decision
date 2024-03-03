from validation import y_pred, y_valid, erreurs_percent, erreurs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.DataFrame()
df["y_pred"] = y_pred
df["y_valid"] = y_valid



vrai_positive = df[(df.y_pred == 0) & (df.y_valid == 0)].size
faux_positive = df[(df.y_pred == 0) & (df.y_valid == 1)].size

faux_negative = df[(df.y_pred == 1) & (df.y_valid == 0)].size
vrai_negative = df[(df.y_pred == 1) & (df.y_valid == 1)].size
print
print(" {:<30}| {:<20}| {:<20}|".format("prediction\\resultats terrain", "positive classe 0", "negative classe 1"))
print(" {:<30}| {:<20}| {:<20}|".format("possitive classe 0", vrai_positive, faux_positive))
print(" {:<30}| {:<20}| {:<20}|".format("possitive classe 1", faux_negative, vrai_negative))

matrice_confusion = np.array([[vrai_positive, faux_positive],[faux_negative, vrai_negative ]])



plt.imshow(matrice_confusion, cmap='YlGnBu', interpolation='nearest')


plt.text(0, 0, f"vrai positive : {matrice_confusion[0, 0]}", ha='center', fontsize=12, va='center', color='red')
plt.text(1, 0, f"faux positive : {matrice_confusion[0, 1]}", ha='center', fontsize=12,  va='center', color='red')
plt.text(0, 1, f"faux negative : {matrice_confusion[1, 0]}", ha='center', fontsize=12, va='center', color='red')
plt.text(1, 1, f"vrai negative : {matrice_confusion[1, 1]}", ha='center', fontsize=12, va='center', color='red')

plt.axis('off')
plt.title(f"matrice de confusion\n taux d'erreurs : {erreurs_percent}")
plt.show()