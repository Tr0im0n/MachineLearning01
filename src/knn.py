import random
import numpy as np
from matplotlib import pyplot as plt

import k_means
import tools


# Tool functions ##########################################################################


def clusters_to_labeled_data(my_data: np.ndarray, clusters: np.ndarray) -> list:
    ans = []
    for i, cluster in enumerate(clusters):
        for index in cluster:
            ans.append([*my_data[index], i])
    return ans


def min_and_max(my_data: np.ndarray) -> tuple[float, float, float, float]:
    x_min = min(my_data[:, 0])
    x_max = max(my_data[:, 0])
    y_min = min(my_data[:, 1])
    y_max = max(my_data[:, 1])
    return x_min, x_max, y_min, y_max


def get_chunk_coord(x: int, y: int, chunk_size: float = 0.3) -> tuple[int, int]:
    return -int((x - 1) // chunk_size), -int((y - 1) // chunk_size)


def construct_spatial_map(my_data: list, chunk_size: float = 0.3) -> dict:
    ans = {}
    for x, y, label in my_data:
        chunk_coord = get_chunk_coord(x, y, chunk_size)
        if chunk_coord not in ans:
            ans[chunk_coord] = [(x, y, label)]
        else:
            ans[chunk_coord].append((x, y, label))
    return ans


def split_data(data, fraction: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    train = []
    test = []
    for i in data:
        if random.random() < fraction:
            train.append(i)
        else:
            test.append(i)
    return np.array(train), np.array(test)


def get_surrounding_chunks(cx, cy, spatial_map, search_radius: int = 1):
    return [(x, y) for x in range(cx - search_radius, cx + search_radius + 1)
                   for y in range(cy - search_radius, cy + search_radius + 1)
                   if (x, y) in spatial_map]


def eval_point(test_point, spatial_map, k: int, search_radius: int = 1):
    # look through the surrounding chunks for points
    # get the distance to the point and its label and add that to the list
    chunk_coord = get_chunk_coord(test_point[0], test_point[1])
    surrounding_chunks = get_surrounding_chunks(chunk_coord[0], chunk_coord[1], spatial_map, search_radius)
    distance_list = []
    for chunk_key in surrounding_chunks:
        for train_x, train_y, label in spatial_map[chunk_key]:
            distance_list.append([tools.distance2(test_point, np.array([train_x, train_y])), label])
    # check if there are k points in this area
    counter = 0
    for distance, label in distance_list:
        if distance >= 0.09:
            counter += 1
    if counter < k:
        print(counter, test_point)
        # recursively try again but with a bigger search radius
        return eval_point(test_point, spatial_map, k, search_radius+1)

    # determine which label is correct
    label_count_dict = {}  # could have been any other type
    max_count = 0
    max_label = -1

    def my_key(ele):
        return ele[0]

    distance_list.sort(key=my_key)  # could lambda be used here?

    label_list = [label for distance, label in distance_list[:k+1]]

    for label in label_list[:k+1]:
        if label not in label_count_dict:
            count = label_list.count(label)
            label_count_dict[label] = count
            if count > max_count:
                max_count = count
                max_label = label

    return max_label


# Final functions ---------------------------------------------------------------------------------

def knn(my_data, k: int = 6):
    train_data, test_data = split_data(my_data)
    my_centers, my_clusters = k_means.k_means(train_data, 5, True)
    labeled_data = clusters_to_labeled_data(train_data, my_clusters)
    spatial_map = construct_spatial_map(labeled_data)

    ans = [[*test_point, eval_point(test_point, spatial_map, k, 1)] for test_point in test_data]

    colors = ['r', 'g', 'b', 'y', 'm', 'c']
    for values in spatial_map.values():
        for x, y, color_i in values:
            plt.scatter(x, y, s=60, c=colors[color_i], marker='*')
    for x, y, color_i in ans:
        plt.scatter(x, y, s=480, c=colors[color_i], marker='+')
    plt.show()

    return ans
