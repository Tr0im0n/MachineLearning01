import os
import numpy as np
import knn


def main() -> None:
    os.chdir(r"..\data")
    file_name = "StarTypeDataset.csv"
    my_data = np.genfromtxt(file_name, delimiter=',', skip_header=1)

    knn_results = knn.knn(my_data)
    print(*knn_results, sep="\n")


if __name__ == "__main__":
    main()




