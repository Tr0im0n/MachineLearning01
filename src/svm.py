import numpy as np
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split
from sklearn import svm


def my_svm(df: pd.DataFrame, results: np.ndarray) -> float:
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(df, results, test_size=0.2, random_state=42)

    # Create an SVM classifier
    clf = svm.SVC(kernel='linear')

    # Train the SVM classifier
    clf.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(x_test)

    # Evaluate the model's accuracy
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    return accuracy
