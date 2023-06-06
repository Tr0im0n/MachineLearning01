import numpy as np


def distance2(p1: np.ndarray, p2: np.ndarray) -> float:
    return sum(tuple(pow(i - j, 2) for i, j in zip(p1, p2)))


def normalize_array(array: np.ndarray) -> np.ndarray:
    return (array - np.mean(array)) / np.std(array)
