import numpy as np
from numpy.linalg import inv, matrix_power

from supervised_learning import SupervisedLearning
class LinearRegression(SupervisedLearning):
    """
        Это строка документирования
        x: feature vector
        Y: target vector
    """
    def __init__(self, feature_vector, target_vector):
        super().__init__(feature_vector, target_vector)
        self.x = feature_vector.values
        self.y = target_vector.values
        self.loss = 'Mean Squared Error'

    # Ordinary least squares
    def find_weights(self):
        x = self.add_bias(self.x)
        y = self.y
        x_t = np.transpose(x)
        x_x_t = x_t @ x
        inv_x_x_t = inv(x_x_t)
        weights = ((inv_x_x_t @ x_t) @ y)

        return weights

    def gradient_descent(self, a=0.0005, iter_cnt = 100):
        w = self.random_weights
        while iter_cnt != 0:
            y_pred = self.add_bias(self.x) @ w
            loss = y_pred-self.y
            gradient = (2/self.n)*(self.add_bias(self.x).T@loss)
            iter_cnt-=1
            w -= a*gradient
        return w


        pass

    def predict(self, w):
        return self.add_bias(self.x)@w

    #def calc_RSS(self):
    #    RSS = matrix_power((self.y-self.predict()), 2)
    #    return RSS


