from typing import Union, Tuple
import numpy as np

import tools


# Tools ---------------------------------------------------------------------------------

def initial_centers(k: int, bounds: tuple) -> np.ndarray:
    means = tuple(b[1]+b[0]/2 for b in bounds)
    radii = tuple(0.3*abs(mean-b[0]) for mean, b in zip(means, bounds))
    return np.array([ [radii[0]*np.cos(i*np.pi), radii[1]*np.sin(i*np.pi)] for i in np.linspace(0, 2, k+1)[:k] ])


def make_new_centers(my_data: np.ndarray, centers: np.ndarray,
                     return_clusters=False) -> np.ndarray | Tuple[np.ndarray, list]:
    # build clusters
    n_centers = len(centers)
    clusters = [[] for _ in range(n_centers)]

    for i, point in enumerate(my_data):
        distances = tuple(tools.distance2(point, center) for center in centers)
        group = np.argmin(distances)
        # Needed this for some reason
        if group > n_centers:
            print(group)
            group = group // n_centers
        # todo why is this wrong???
        clusters[group].append(i)

    # get new centers from clusters
    ans = np.zeros_like(centers)
    for n, cluster in enumerate(clusters):
        my_sum = np.array([0.0, 0.0])
        for i in cluster:
            my_sum += my_data[i]
        ans[n] = my_sum/(len(cluster)-1)

    if return_clusters:
        return ans, clusters

    return ans


# Final functions -----------------------------------------------------------------

def k_means(my_data: np.ndarray, k: int = 5, clusters: bool = False):
    bounds = tuple((min(my_data[:, i]), max(my_data[:, i])) for i in range(my_data.shape[1]))
    centers = initial_centers(k, bounds)

    for _ in range(7):
        centers = make_new_centers(my_data, centers)

    return make_new_centers(my_data, centers, clusters)
