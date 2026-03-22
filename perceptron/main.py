import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from perceptron import Perceptron


def main():
    X_full, y_full = datasets.load_iris(return_X_y=True)

    # Iris labels: 0=Setosa, 1=Versicolor, 2=Virginica
    # For binary classification, we will only use Setosa (0) and Versicolor (1). We will ignore Virginica (2).
    X = X_full[y_full != 2]
    y = y_full[y_full != 2]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Perceptron(leaning_rate=0.01, n_iters=1000)
    model.fit(np.array(X_train), np.array(y_train))

    predictions = model.predict(np.array(X_test))
    print(f"Predictions: {predictions}")
    accuracy = np.mean(predictions == y_test)
    print(f"Actual labels: {y_test}")
    print(f"Perceptron Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
