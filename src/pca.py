import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def sort_val_vec(values: np.ndarray, matrix: np.ndarray) -> (np.ndarray, np.ndarray):
    order = np.argsort(values)[::-1]
    return [values[i] for i in order], [matrix[i] for i in order]


def pca(df: pd.DataFrame) -> (np.ndarray, np.ndarray):
    cov = df.cov()
    e_vals, e_vecs = np.linalg.eig(cov)
    e_vecs = np.transpose(e_vecs)
    return sort_val_vec(e_vals, e_vecs)


def pca_table(df: pd.DataFrame, show_scree: bool = True) -> None:
    vals, vecs = pca(df)
    print("Eigenvalue", end="")
    for val in vals:
        print(f" & {val:.2f}", end="")
    print(" \\\\")

    sum_val = sum(vals)
    fracs = tuple(100 * val / sum_val for val in vals)
    print("Percentage(%)", end="")
    for frac in fracs:
        print(f" & {frac:.2f}", end="")
    print(" \\\\")

    for i, attribute in enumerate(df.columns):
        print(attribute, end="")
        for vec in vecs:
            print(f" & {round(1000*vec[i])}", end="")
        print(" \\\\")

    if show_scree:
        plt.plot(range(1, 12), fracs, color='b')
        for i, frac in enumerate(fracs):
            plt.scatter(i+1, frac, color='b', label=f"c{i+1}: {frac:.2f}%")
        plt.title("Scree plot of PCA")
        plt.legend()
        plt.xlabel("Principal Component")
        plt.ylabel("Variance explained")
        plt.xticks(range(1, 12))
        plt.show()
