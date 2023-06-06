
import os
import pandas as pd

import df_cleaning
import df_visualization
import pca
import neural_net


def upper_limit_dict() -> dict:
    return {'fixed acidity': 14.5,
            'volatile acidity': 1.15,
            'citric acid': 0.8,
            'residual sugar': 10,
            'chlorides': 0.5,
            'free sulfur dioxide': 60,
            'total sulfur dioxide': 200,
            'density': None,
            'pH': 3.8,
            'sulphates': 1.5,
            'alcohol': 14.5}


def weights(n: int = 12):
    if n == 12:
        return (5.6674, 0.1000, -0.3074, 0.1839, 0.0253, -0.0934,
                -0.0461, -0.1725, -0.1481, -0.0456, 0.2475, 0.3995)
    elif n == 10:
        return (5.6674, -0.0644, 0.2711, -0.2079,  0.0266,
                -0.0771, -0.0098, -0.1128, 0.0844, 0.1159)
    elif n == 8:    # ml slopes
        return (5.6683, -0.0542, 0.2828, -0.1953, 0.0282,
                -0.0855, -0.0092, -0.0991, 0.0689, 0.1269)
    print(f"{n} not in library")


def main() -> None:
    # os and reads --------------------------------------------------------------------------
    os.chdir(r"..\data")
    file_name = "winequality-red.csv"
    initial_wine_df = pd.read_csv(file_name, delimiter=';')

    # data pre-processing ---------------------------------------------------------------------
    clipped_wine_df = df_cleaning.remove_high_outliers(initial_wine_df, upper_limit_dict())
    clipped_qualities_array = clipped_wine_df["quality"].values
    normalized_wine_df = df_cleaning.normalize_df(clipped_wine_df.iloc[:, :11])
    dimension2_reduced_wine_df = df_cleaning.remove_n_dimensions(normalized_wine_df, 2, return_df=True)
    # pca.pca_table(normalized_wine_df)

    # data visualization -------------------------------------------------------------------
    # df_visualization.sort_attributes(clipped_wine_df)
    # df_visualization.plot_against_quality(normalized_wine_df, clipped_qualities_array)
    # df_visualization.plot_against_quality(dimension2_reduced_wine_df, clipped_qualities_array)
    # df_visualization.scatter_2_components(dimension2_reduced_wine_df, clipped_qualities_array)
    # df_visualization.plot_real_vs_predicted(normalized_wine_df, clipped_qualities_array, weights(12))
    # df_visualization.plot_real_vs_predicted(dimension2_reduced_wine_df, clipped_qualities_array, weights(10))

    # machine learning --------------------------------------------------------------------------
    ans = neural_net.regression(dimension2_reduced_wine_df, clipped_qualities_array, hidden_layer=False, epochs=256)
    loss = f"{ans[2]:.4f}"
    ml_weights = float(ans[1]), *tuple(float(i) for i in ans[0])
    print(*tuple(f"{i:.4f}" for i in ml_weights))
    df_visualization.plot_real_vs_predicted(dimension2_reduced_wine_df, clipped_qualities_array, ml_weights,
                                            testing_loss=loss)

    # neural_net.classification(dimension2_reduced_wine_df, clipped_qualities_array)


if __name__ == "__main__":
    main()
