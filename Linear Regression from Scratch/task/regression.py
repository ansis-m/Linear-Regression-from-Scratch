
import numpy as np

x = np.array([4.0, 4.5, 5, 5.5, 6.0, 6.5, 7.0])
y = np.array([33, 42, 45, 51, 53, 61, 62])


class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):

        self.fit_intercept = fit_intercept
        self.coefficients = ...
        self.coefficient = ...
        self.intercept = ...
        self.map = {}

    def fit(self, X=x, Y=y):

        if self.fit_intercept:
            X = np.vstack([x, np.ones(len(x))]).T
        else:
            X = X.T
        self.coefficients, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        self.coefficient, self.intercept = self.coefficients
        self.map = {'Intercept': self.intercept, 'Coefficient': self.coefficients[:-1]}
        return self

    def print(self):
        print(self.map)


def main():
    CustomLinearRegression().fit().print()


if __name__ == "__main__":
    main()
