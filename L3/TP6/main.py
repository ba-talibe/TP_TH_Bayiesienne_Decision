from tools import *
import numpy as np
import seaborn as sns
from utils import plot_decision
from train_tools import bayesienne_decision
from data_loader import train_data, validation_data, train_data2, validation_data2
from fonction_decision import get_vector_predictor, get_predictor


def exec(train_data, validation_data):
    show(train_data)
    plt.show()
    x1_min = train_data['x1'].min()
    x2_min = train_data['x2'].min()
    x1_max = train_data['x1'].max()
    x2_max = train_data['x2'].max()
    param = bayesienne_decision(train_data)
    sns.heatmap(param[1], annot=True)
    plt.show()
    sns.heatmap(param[3], annot=True)
    plt.show()
    for par in param:
        print(par)
    show(validation_data)
    plot_decision(x1_min,x1_max,x2_min,x2_max, get_predictor(*param))
    plt.show()  
    validate(validation_data, *param)

if __name__ == '__main__':
    exec(train_data, validation_data)
    exec(train_data2, validation_data2)
