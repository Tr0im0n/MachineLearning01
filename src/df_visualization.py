import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def sort_attributes(df: pd.DataFrame) -> None:
    high_vals = ['free sulfur dioxide', 'total sulfur dioxide']
    medium_vals = ['fixed acidity', 'residual sugar', 'pH', 'alcohol', 'quality']
    low_vals = ['volatile acidity', 'citric acid', 'chlorides', 'density', 'sulphates']
    titles = ["Low Vals", "Medium Vals", "High Vals"]
    all_vals = [low_vals, medium_vals, high_vals]
    maxes = [2, 15, 280]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle("Sorted attributes")

    for column in df.columns:
        ax1.plot(np.sort(df[column]), label=column)
    ax1.legend()
    ax1.set_title("All Vals")

    for title, maxi, vals, ax in zip(titles, maxes, all_vals, (ax2, ax3, ax4)):
        for name in vals:
            ax.plot(np.sort(df[name]), label=name)
        ax.set_title(title)
        ax.legend()
    plt.show()


def plot_against_quality(df: pd.DataFrame, qualities: np.ndarray, *,
                         show_separate: bool = False) -> None:
    n_components = df.shape[1]
    n_cols = (n_components+1)//2
    column_type = "Components" if n_components != 11 else "Attributes"

    fig, axs = plt.subplots(2, n_cols, sharey="all")
    fig.suptitle(f"{column_type} vs quality, with linfit line")
    for i, attribute in enumerate(df.columns):
        ax = axs[i // n_cols][i % n_cols]
        ax.scatter(df[attribute], qualities)
        ax.set_xlabel(attribute)
        if not i % n_cols:
            ax.set_ylabel('quality')
        coef = np.polyfit(df[attribute], qualities, 1)
        min_at = min(df[attribute])
        max_at = max(df[attribute])
        xs = np.linspace(min_at, max_at, 2)
        ys = np.polyval(coef, xs)
        ax.plot(xs, ys, color='r', label=f"slope: {coef[0]:.2f}")
        ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.92))
        print(f"{coef[0]:.4f}")
    plt.show()

    if show_separate:
        for column in df.columns:
            plt.scatter(df[column], qualities, marker="D")
            plt.xlabel(column)
            plt.ylabel('quality')
            plt.show()


def scatter_2_components(df: pd.DataFrame, qualities: np.ndarray, components: tuple[int, int] = (2, 3), *,
                         average_quality: float = 5.5, base_size: int = 32,
                         min_score: int = 3, max_score: int = 8) -> None:
    colors = ['k', 'k', 'k', 'r', 'g', 'b', 'y', 'm', 'c', 'k']
    color_list = [colors[q] for q in qualities]
    size_list = [base_size*(math.floor(abs(q-average_quality))+1) for q in qualities]
    plt.scatter(df[f"component {components[0]}"], df[f"component {components[1]}"], s=size_list, c=color_list)
    for i in range(min_score, max_score+1):
        plt.scatter(-4, -4, c=colors[i], label=i)
    plt.scatter(-4, -4, s=72, c='w')
    plt.title("Scatter plot of 2 principal components with quality as color")
    plt.xlabel(f"Component {components[0]}")
    plt.ylabel(f"Component {components[1]}")
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.show()


def plot_real_vs_predicted(df: pd.DataFrame, qualities: np.ndarray, weights: tuple[float, ...], *,
                           testing_loss: any = "?", min_score: int = 3, max_score: int = 8) -> None:

    predicted_qualities = np.dot(df, np.transpose(weights[1:])) + weights[0]
    plt.scatter(qualities, predicted_qualities, marker="D", label="scatter")
    xs = [min_score, max_score]
    plt.plot(xs, xs, color='k', label="y=x")
    coef = np.polyfit(qualities, predicted_qualities, 1)
    plt.plot(xs, np.polyval(coef, xs), color='r', label=f"fit, testing loss: {testing_loss}")

    input_type = "dimension reduced" if df.shape[1] < 11 else "normalised"
    weights_source = "machine learning" if testing_loss != "?" else "lin fit"

    plt.title(f"real vs predicted qualities, of {weights_source} on {input_type} input")
    plt.gca().set_aspect("equal")
    plt.ylabel("predicted qualities")
    plt.xlabel("real qualities")
    plt.legend()
    plt.show()
