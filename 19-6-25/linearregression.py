import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class linearRegression:
    def _init_(self):
        self.b0 , self.b1 = 0,0

    def fit (self, X , y):
        X_mean = np.mean(X)
        y_mean = np.mean(y)
        ssxy , ssx = 0, 0
        for _ in range (len (X)) :
            ssxy += (X[]-X_mean)*(y[]-y_mean)
            ssx += (X[_]-X_mean)**2
        self.b1 = ssxy / ssx
        self.b0 =y_mean - (self.b1 * X_mean)
        return self.b0,self.b1    
    
    def predict (self,Xi) :
        y_hat = self.b0 + (self.b1 * Xi)
        return np.squeeze(y_hat)
    
    def MEAN_SQUARD_ERROR(self):
        error=self.y-self.y_hat
        squared_error=error**2
        return np.mean(squared_error)

    def gradientDescent(self, alpha=0.00005, epchos=1):
        error=self.y-self.y_hat
        n=len(self.X)
        for _ in range(epchos):
            del_b1=(-2/n)*np.sum(self.X*error)
            del_b0=(-2/n)*np.sum(error)
            self.b1=self.b1-alpha*del_b1
            self.b0=self.b0-alpha*del_b0

if _name_ =='_main_':
    heights = np.array([
        [160],[171],[182],[180],[154]
    ])

    weights = np.array ([
        72,76,77,83,76
    ])    

    lr = linearRegression (input_data=heights, input_label=weights)
    b0,b1 = lr.fit (X=heights, y= weights)

    Xi = 176
    y_hat = lr.predict(Xi=heights)
    
    print(f'True label :{weights}')
    print(f'Predicted label: {y_hat}')

    def meanSquareError (y_true, y_pred):
        error = y_true - y_pred
        squaredError = error ** 2
        return np.mean (squaredError)
    
    mse = meanSquareError (y_true=weights , y_pred=y_hat)
    print (f'Hardcodded function:{mse}')

    skmse = mean_squared_error(weights,y_hat)
    print(f'Sklearn MSE:{skmse}')