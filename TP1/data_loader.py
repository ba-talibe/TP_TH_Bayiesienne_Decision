import pandas as pd 
import numpy as np 


train_data = pd.read_csv("TP1/tp1_data/tp1_data_train.txt", names=['x', 'y'])
validation_data = pd.read_csv("TP1/tp1_data/tp1_data_valid.txt", names=["x", "y"])



X_train = train_data.x.values
y_train = train_data.y.values

X_valid = validation_data.x.values
y_valid = validation_data.y.values

if __name__ == '__main__':
    print("cardinal classe 1", np.sum(y_train))
    print("cardinal classe 0", y_train.size - np.sum(y_train))
    print("[+] Done")