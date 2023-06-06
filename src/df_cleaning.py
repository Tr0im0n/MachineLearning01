import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import pca


def remove_high_outliers(df: pd.DataFrame, upper_limit_dict: dict) -> pd.DataFrame:
    high_outliers_indices = []
    for key, value in upper_limit_dict.items():
        indices = df[df[key] > value].index
        for i in indices:
            if i not in high_outliers_indices:
                high_outliers_indices.append(i)
    return df.drop(high_outliers_indices)


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize all columns using StandardScaler
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df)
    # Create a new DataFrame with normalized values
    return pd.DataFrame(normalized_data, columns=df.columns)


def remove_n_dimensions(df: pd.DataFrame, n: int = 2, return_df: bool = False) -> np.ndarray | pd.DataFrame:
    vals, vecs = pca.pca(df)
    transform_matrix = np.transpose(vecs[:-n])
    dimension_reduced_array = np.dot(df, transform_matrix)
    if not return_df:
        return dimension_reduced_array
    else:
        df = pd.DataFrame(dimension_reduced_array)
        df.columns = [f"component {i+1}" for i in range(len(vals)-n)]
        return df
