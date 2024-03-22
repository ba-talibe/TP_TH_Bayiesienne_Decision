from tools import *
import numpy as np
from utils import plot_decision
from train_tools import bayesienne_decision
from data_loader import train_data, validation_data
from fonction_decision import get_vector_predictor, get_predictor


if __name__ == '__main__':
    show(train_data)
    plt.show()
    x1_min = train_data['x1'].min()
    x2_min = train_data['x2'].min()
    x1_max = train_data['x1'].max()
    x2_max = train_data['x2'].max()
    param = bayesienne_decision(train_data)
    prediction = get_vector_predictor(*param)
    show(validation_data)
    plot_decision(x1_min,x1_max,x2_min,x2_max, get_predictor(*param))
    plt.show()  
    validate(validation_data, *param)
