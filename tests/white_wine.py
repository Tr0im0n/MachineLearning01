
import os
import pandas as pd

import df_cleaning
import df_visualization
import neural_net
import pca


def upper_limit_dict() -> dict:
    return {'fixed acidity': 11,
            'volatile acidity': 0.8,
            'citric acid': 1.1,
            'residual sugar': 21.5,
            'chlorides': 0.22,
            'free sulfur dioxide': 150,
            'total sulfur dioxide': 264,
            'density': 1.005,
            'pH': None,
            'sulphates': 1.02,
            'alcohol': None}


def weights(n: int = 12):
    if n == 12:
        return (5.6674, 0.1000, -0.3074, 0.1839, 0.0253, -0.0934,
                -0.0461, -0.1725, -0.1481, -0.0456, 0.2475, 0.3995)
    elif n == 11:
        return (5.6674, -0.0644, 0.2711, -0.2079,  0.0266,
                -0.0771, -0.0098, -0.1128, 0.0844, 0.1159)
    elif n == 9:
        return
    print(f"{n} not in library")


def main() -> None:
    # os and reads --------------------------------------------------------------------------
    os.chdir(r"..\data")
    file_name = "winequality-white.csv"
    initial_wine_df = pd.read_csv(file_name, delimiter=';')

    # data pre-processing ---------------------------------------------------------------------
    clipped_wine_df = df_cleaning.remove_high_outliers(initial_wine_df, upper_limit_dict())
    clipped_qualities_array = clipped_wine_df["quality"].values
    normalized_wine_df = df_cleaning.normalize_df(clipped_wine_df.iloc[:, :11])
    # dimension1_reduced_wine_df = df_cleaning.remove_n_dimensions(normalized_wine_df, 1, return_df=True)
    dimension3_reduced_wine_df = df_cleaning.remove_n_dimensions(normalized_wine_df, 3, return_df=True)
    # pca.pca_table(normalized_wine_df)

    # data visualization -------------------------------------------------------------------
    # df_visualization.sort_attributes(clipped_wine_df)
    # df_visualization.plot_against_quality(clipped_wine_df, clipped_qualities_array)
    # df_visualization.plot_against_quality(dimension1_reduced_wine_df, clipped_qualities_array)
    # df_visualization.scatter_2_components(dimension3_reduced_wine_df, clipped_qualities_array, (1, 3),
    #                                       average_quality=6, base_size=16, max_score=9)
    # df_visualization.plot_lin_fit_predicted(normalized_wine_df, clipped_qualities_array)
    # df_visualization.plot_lin_fit_predicted(dimension3_reduced_wine_df, clipped_qualities_array)

    # machine learning --------------------------------------------------------------------------
    # neural_net.regression(dimension3_reduced_wine_df, clipped_qualities_array, hidden_layer=False)
    # neural_net.classification(dimension3_reduced_wine_df, clipped_qualities_array)


if __name__ == "__main__":
    main()
