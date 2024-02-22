import matplotlib.pyplot as plt
from data_loader import train_data, X_train, y_train


classe0 = train_data[train_data.y == 0]
classe01 = train_data[train_data.y == 1]

plt.figure()
plt.hist(classe0.x, alpha=.5)
plt.hist(classe01.x, alpha=0.7)
plt.show()
