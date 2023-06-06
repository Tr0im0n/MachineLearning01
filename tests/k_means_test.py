import os
import numpy as np
import k_means
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def anim(my_data: np.ndarray, k=5):
    fig = plt.figure()
    ax = plt.axes()

    bounds = tuple((min(my_data[:, i]), max(my_data[:, i])) for i in range(my_data.shape[1]))
    centers = k_means.initial_centers(k, bounds)

    def init():
        plt.scatter(my_data[:, 0], my_data[:, 1], s=36, marker='*')
        plt.scatter(*centers.transpose(), s=48, c='k')
        return

    def animate(frame):
        ax.clear()
        ax.set_title(f"Frame: {frame}")

        nonlocal centers

        if not frame:
            centers = k_means.initial_centers(k, bounds)
            init()
            return

        centers, clusters = k_means.make_new_centers(my_data, centers, True)
        for cluster in clusters:
            plt.scatter(my_data[cluster, 0], my_data[cluster, 1], s=36, marker='*')

        plt.scatter(*centers.transpose(), s=48, c='k')
        return

    animation = FuncAnimation(fig,
                              func=animate,
                              frames=8,
                              init_func=init,
                              interval=500,
                              repeat=True)
    plt.show()
    plt.close()
    return


def main() -> None:
    os.chdir(r"..\data")
    file_name = "StarTypeDataset.csv"
    my_data = np.genfromtxt(file_name, delimiter=',', skip_header=1)
    k_means_results = k_means.k_means(my_data, 5, True)
    print(*tuple(f"Center: {i} \t Cluster indices: {j}" for i, j in zip(*k_means_results)), sep="\n")
    anim(my_data, 5)


if __name__ == "__main__":
    main()
