
import numpy as np

x = [4, 4.5, 5, 5.5, 6, 6.5, 7]
w = [1, -3, 2, 5, 0, 3, 6]
z = [11, 15, 12, 9, 18, 13, 16]
y = [33, 42, 45, 51, 53, 61, 62]


class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):

        self.fit_intercept = fit_intercept
        self.coefficients = ...
        self.y_pred = ...

    def fit(self):

        data = np.column_stack([x, w, z])
        self.coefficients, _, _, _ = np.linalg.lstsq(data, y, rcond=None)
        self.y_pred = np.dot(data, self.coefficients)
        return self

    def print(self):
        print(self.y_pred)


def main():
    CustomLinearRegression().fit().print()


if __name__ == "__main__":
    main()
