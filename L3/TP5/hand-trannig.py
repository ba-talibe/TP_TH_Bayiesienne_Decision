from  tools import show, validate
from utils import plot_decision
from data_loader import train_data, validation_data
from data_loader import train_data2, validation_data2
from matplotlib import pyplot as plt

from fonction_decision import get_predictor




if __name__ == '__main__':
    show(train_data2)
    plt.show()
    x_min = input("entre x1 et x2 min:").split(' ')
    x_min = [float(x) for x in x_min]
    x_max = input("entre x1 et x2 max:").split(' ')
    x_max = [float(x) for x in x_max]
    prediction = get_predictor(x_min, x_max)
    show(train_data2)
    plot_decision(x_min[0],x_max[0],x_min[1],x_max[1], prediction)
    plt.show()
    validate(validation_data2, x_min, x_max)
