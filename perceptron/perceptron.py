from typing import Optional

import numpy as np


class Perceptron:
    def __init__(self, leaning_rate: float = 0.01, n_iters: int = 1000) -> None:
        self.leaning_rate: float = leaning_rate
        self.n_iters: int = n_iters
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[float] = None

    def weighted_sum(self, X: np.ndarray) -> np.ndarray | float:
        """
        Calculate the weighted sum

        Parameters
        ----------
        X : np.ndarray. During the training phase, X is a 1D numpy array representing a single sample. During the prediction phase, X is a 2D numpy array representing multiple samples.

        Returns
        -------
        np.ndarray | float. During the training phase, it returns a float. During the prediction phase, it returns a numpy array of floats.
        """
        if self.weights is None or self.bias is None:
            raise ValueError(
                "Weights and bias must be initialized before calculating the weighted sum."
            )
        return np.dot(X, self.weights) + self.bias

    def activation_function(self, x: np.ndarray | float) -> np.ndarray | int:
        """
        Activation function

        Parameters
        ----------
        x : np.ndarray | float. During the training phase, x is a float. During the prediction phase, x is a numpy array.

        Returns
        -------
        np.ndarray | int. During the training phase, it returns an int (0 or 1). During the prediction phase, it returns a numpy array of 0s and 1s.
        """
        return np.where(x >= 0, 1, 0)

    def clean_labels(self, y: np.ndarray) -> np.ndarray:
        """
        Enforce binary labels (0 and 1)
        """
        numberOfLabelsCleaned = np.sum((y != 0) & (y != 1))
        if numberOfLabelsCleaned > 0:
            print(
                f"Expecting only 0 and 1 labels. Cleaning {numberOfLabelsCleaned} labels."
            )
        return np.where(
            y > 0, 1, 0
        )  # equivalent to np.array([1 if i > 0 else 0 for i in y])

    def compute_error(self, y_predicted: int, y_expected: int) -> int:
        return y_expected - y_predicted

    def update_weights_and_bias(self, error: int, X: np.ndarray) -> None:
        if self.weights is None or self.bias is None:
            raise ValueError("Unitialized weights and bias can't be updated")
        update = self.leaning_rate * error
        self.weights += update * X
        self.bias += update

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the training data

        Parameters
        ----------
        X : np.ndarray. The feature matrix
        y : np.ndarray. The labels
        """

        # Initializing weights and bias

        # shape returns a tuple (number of samples, number of features)
        numberOfFeatures = X.shape[1]
        self.weights = np.zeros(numberOfFeatures)
        self.bias = 0.0

        # Clean labels to ensure they are binary
        cleaned_y = self.clean_labels(y)

        for _ in range(self.n_iters):
            for index, x_i in enumerate(X):
                linear_output = self.weighted_sum(x_i)
                y_predicted = self.activation_function(linear_output)

                # Update weights and bias
                error = self.compute_error(int(y_predicted), cleaned_y[index])
                self.update_weights_and_bias(error, x_i)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the given feature matrix

        Parameters
        ----------
        X : np.ndarray. The feature matrix
        """
        if self.weights is None or self.bias is None:
            raise ValueError("The perceptron must be trained before calling predict.")

        linear_outputs = self.weighted_sum(X)
        return np.asarray(self.activation_function(linear_outputs))
