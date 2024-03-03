import matplotlib.pyplot as plt
from data_loader import train_data, X_train, y_train


classe0 = train_data[train_data.y == 0]
classe1 = train_data[train_data.y == 1]

plt.figure()
plt.hist(classe0.x, alpha=.5, label=f"classe {0} : {classe0.shape[0]} ")
plt.hist(classe1.x, alpha=0.7, label=f"classe {1}: {classe1.shape[0]}")
plt.legend()
plt.title("tp1_data_train.txt")
plt.show()
