
import numpy as np

capacity = [0.9, 0.5, 1.75, 2.0, 1.4, 1.5, 3.0, 1.1, 2.6, 1.9]
age = [11, 11, 9, 8, 7, 7, 6, 5, 5, 4]
cost_per_ton = [21.95, 27.18, 16.9, 15.37, 16.03, 18.15, 14.22, 18.72, 15.4, 14.69]


class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):

        self.fit_intercept = fit_intercept
        self.coefficients = ...
        self.cost_per_ton_pred = ...
        self.map = {}

    def fit(self):

        data = np.column_stack([np.ones(len(capacity)), capacity, age])
        self.coefficients, _, _, _ = np.linalg.lstsq(data, cost_per_ton, rcond=None)
        self.cost_per_ton_pred = np.dot(data, self.coefficients)

        sst = sum((np.array(cost_per_ton) - np.mean(cost_per_ton))**2)
        sse = np.sum((self.cost_per_ton_pred - np.array(cost_per_ton))**2)

        self.map['Intercept'] = self.coefficients[0]
        self.map['Coefficient'] = self.coefficients[1:]
        self.map['R2'] = 1 - sse/sst
        self.map['RMSE'] = np.sqrt(sse/len(cost_per_ton))
        return self

    def print(self):
        print(self.map)


def main():
    CustomLinearRegression().fit().print()


if __name__ == "__main__":
    main()
