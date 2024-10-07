from tools import show, validate


if __name__ == '__main__':
    pathdir = "TP4/tp4_data/"
    common_name = "tp2_data"
    train_data_file = "tp4_data1_train.txt"
    valid_data_file = "tp4_data1_valid.txt"

    show(train_data_file, filedir=pathdir, title="choisir delta 1")
    delta1 = float(input("enter delta 1 choisi : "))
    show(train_data_file, filedir=pathdir, title="choisir delta 2")
    delta2 = float(input("enter delta 2 choisi : "))
    delta = [delta1, delta2]
    erreurs, matrice_confusion = validate(valid_data_file, delta, filedir=pathdir)