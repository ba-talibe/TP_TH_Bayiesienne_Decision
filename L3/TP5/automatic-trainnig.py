from  tools import show, validate
from utils import plot_decision
from data_loader import *
from matplotlib import pyplot as plt
from train_tools import train_perceptron
from fonction_decision import get_predictor




if __name__ == '__main__':
    show(train_data)
    plt.show()
    # x_min = input("entre x1 et x2 min:").split(' ')
    # x_min = [float(x) for x in x_min]p
    # x_max = input("entre x1 et x2 max:").split(' ')
    # x_max = [float(x) for x in x_max]
    x_min, x_max = train_perceptron(train_data, iters=100)
    
    print(x_min, x_max)
    prediction = get_predictor(x_min, x_max)
    show(validation_data)
    plot_decision(x_min[0],x_max[0],x_min[1],x_max[1], prediction)
    plt.show()  
    validate(validation_data, x_min, x_max)
