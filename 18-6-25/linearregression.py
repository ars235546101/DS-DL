import numpy as np

class linearregresion:
    def __init__(self):
        self.b0, self.b1 = 0, 0

    def fit(self, X, y):
        Xmean = np.mean(X)
        ymean = np.mean(y)
        ssxy, ssx = 0, 0
        for i in range(len(X)):
            ssxy += (X[i] - Xmean) * (y[i] - ymean)
            ssx += (X[i] - Xmean) ** 2
        self.b1 = ssxy / ssx
        self.b0 = ymean - (self.b1 * Xmean)

    def predict(self, X):
        Y = self.b0 + self.b1 * X
        print('The value of Y is', Y)
        return Y

if __name__ == '__main__':
    heights = np.array([160, 171, 182, 180, 154])
    weights = np.array([72, 76, 77, 83, 76])

    print(f'The shape of heights is {heights.shape}')
    print(f'The shape of weights is {weights.shape}')

    model = linearregresion()
    model.fit(heights, weights)
    model.predict(heights)
    model.predict(176)

    print("b0 (intercept):", model.b0)
    print("b1 (slope):", model.b1)

