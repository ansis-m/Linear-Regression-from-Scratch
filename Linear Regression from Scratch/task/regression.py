
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

f1 = [2.31, 7.07, 7.07, 2.18, 2.18, 2.18, 7.87, 7.87, 7.87, 7.87]
f2 = [65.2, 78.9, 61.1, 45.8, 54.2, 58.7, 96.1, 100.0, 85.9, 94.3]
f3 = [15.3, 17.8, 17.8, 18.7, 18.7, 18.7, 15.2, 15.2, 15.2, 15.2]
y = [24.0, 21.6, 34.7, 33.4, 36.2, 28.7, 27.1, 16.5, 18.9, 15.0]


class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):

        self.fit_intercept = fit_intercept
        self.coefficients = ...
        self.y_pred = ...
        self.map = {}

    def fit(self):

        data = np.column_stack([np.ones(len(f1)), f1, f2, f3])
        self.coefficients, _, _, _ = np.linalg.lstsq(data, y, rcond=None)
        self.y_pred = np.dot(data, self.coefficients)

        sst = sum((np.array(y) - np.mean(y))**2)
        sse = np.sum((self.y_pred - np.array(y))**2)


        regression = LinearRegression(fit_intercept=True)
        regression.fit(np.column_stack([f1, f2, f3]), y)
        # print(regression.intercept_)
        # print(regression.coef_)
        r_predict = regression.predict(np.column_stack([f1, f2, f3]))

        r_sse = np.sum((r_predict - np.array(y))**2)



        self.map['Intercept'] = self.coefficients[0] - regression.intercept_
        self.map['Coefficient'] = self.coefficients[1:] - regression.coef_
        self.map['R2'] = (r_sse - sse)/sst
        self.map['RMSE'] = np.sqrt(sse/len(y)) - np.sqrt(r_sse/len(y))
        return self

    def print(self):
        print(self.map)


def main():
    CustomLinearRegression().fit().print()


if __name__ == "__main__":
    main()
