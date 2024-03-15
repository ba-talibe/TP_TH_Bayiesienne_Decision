import os
from tools import test, show, valid
from train_tools import get_optimal_boundary
from pathlib import Path



if __name__ == '__main__':
    pathdir = "TP3/tp2_data/"
    common_name = "tp2_data"

    files = Path(pathdir).glob(f"{common_name}*")
    files = list(files)
    for i in range(1, len(files)//2 + 1):
        print(f"{common_name}{i}*")
        dataset_files = list(Path(pathdir).glob(f"{common_name}{i}*"))
        dataset_files = [str(file) for file in dataset_files]

        traindata_file = [file for file in dataset_files if "train" in file][0]
        validdata_file = [file for file in dataset_files if "valid" in file][0]

        show(traindata_file, filedir="")
        delta = get_optimal_boundary(traindata_file, filedir="")
        erreurs, matrice_confusion = valid(validdata_file, delta, filedir="")

        print("\n\n")