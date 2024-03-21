import pandas as pd 
import numpy as np 


train_data = pd.read_csv("TP5/tp5_data/tp5_data1_train.txt", names=["x1", 'x2', 'y'])
validation_data = pd.read_csv("TP5/tp5_data/tp5_data1_valid.txt", names=["x1", "x2", "y"])

train_data2 = pd.read_csv("TP5/tp5_data/tp5_data2_train.txt", names=["x1", 'x2', 'y'])
validation_data2 = pd.read_csv("TP5/tp5_data/tp5_data2_valid.txt", names=["x1", "x2", "y"])


X_train = train_data[["x1", "x2"]].values

y_train = train_data.y.values


X_train2 = train_data[["x1", "x2"]].values

y_train2 = train_data.y.values

X_valid = validation_data[["x1", "x2"]].values
y_valid = validation_data.y.values

X_valid2 = validation_data[["x1", "x2"]].values
y_valid2 = validation_data.y.values

if __name__ == '__main__':
    # print("cardinal classe 1", np.sum(y_train))
    # print("cardinal classe 0", y_train.size - np.sum(y_train))
    # print("[+] Done")

    print(train_data)