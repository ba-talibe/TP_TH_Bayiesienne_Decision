from tools import show, validate
from train_tools import get_optimal_boundary

if __name__ == '__main__':
    pathdir = "TP4/tp4_data/"
    common_name = "tp2_data"
    train_data_file = "tp4_data1_train.txt"
    valid_data_file = "tp4_data1_valid.txt"

    show(train_data_file, filedir=pathdir, save=True)
    delta = get_optimal_boundary(train_data_file, filedir=pathdir)
    show(train_data_file, filedir=pathdir, delta=delta)
    erreurs, matrice_confusion =  validate(valid_data_file, delta, filedir=pathdir)