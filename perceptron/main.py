import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from perceptron import Perceptron


def main():
    X_full, y_full = datasets.load_iris(return_X_y=True)

    # Iris labels: 0=Setosa, 1=Versicolor, 2=Virginica
    # Iris Setosa is perfectly linearly separable from both Iris Versicolor and Iris Virginica.
    # Perceptron accuracy can be 100% with those classes.
    #
    # Versicolor and Virginica classes are not linearly separable.
    # Perceptron accuracy will be less than 100% with those classes.
    mask = y_full != 0
    X = X_full[mask]
    y = (y_full[mask] == 2).astype(int)  # Versicolor=0, Virginica=1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2
    )

    model = Perceptron(learning_rate=0.1, n_iters=200)
    model.fit(np.array(X_train), np.array(y_train))

    predictions = model.predict(np.array(X_test))
    print(f"Predictions: {predictions}")
    accuracy = np.mean(predictions == y_test)
    print(f"Actual labels: {y_test}")
    print(f"Perceptron Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
