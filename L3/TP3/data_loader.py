import pandas as pd 
import numpy as np 


train_data = pd.read_csv("TP3/tp1_data/tp1_data_train.txt")
valid_data = pd.read_csv("TP3/tp1_data/tp1_data_valid.txt")

train_data.columns = ['x', 'y']
valid_data.columns = ["x", "y"]

X_train = train_data.x.values
y_train = train_data.y.values

X_valid = valid_data.x.values
y_valid = valid_data.y.values

if __name__ == '__main__':
    print("cardinal classe 1", np.sum(y_train))
    print("cardinal classe 0", y_train.size - np.sum(y_train))
    print("[+] Done")

