import os 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




def show(df, save=True, title=None, delta =None):
    
    classes = df.y.unique()
    marker = ["+", "x"]
    repartitons = {classe : df[df.y == classe] for classe in classes}
    
    plt.figure(figsize=(10, 8))
    
    for i, classe in enumerate(classes):
        card = repartitons[classe].size
        plt.scatter(repartitons[classe].x1, repartitons[classe].x2, marker=marker[i], alpha=.7, label=f"{str(classe)}, {card}")


    if title is not None:
        plt.title(title)

    if delta is not None:
        for deltai in delta:
            plt.axvline(deltai )
            plt.text(deltai, 2, f"{np.round(deltai)}")

    plt.axis("equal")
    plt.legend()
    plt.show()
