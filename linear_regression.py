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
        self.random_weights = np.random.default_rng(37).normal(0.5, 0.1, (self.m+1, 1)) # плюс 1 для bais

    # Ordinary least squares
    def ols(self):
        x = self.add_bias(self.x)
        y = self.y
        x_t = np.transpose(x)
        x_x_t = x_t @ x
        inv_x_x_t = inv(x_x_t)
        weights = ((inv_x_x_t @ x_t) @ y)

        return weights

    def gradient_descent(self, a=0.0005, iter_cnt = 100):
        w = self.random_weights
        x = self.add_bias(self.x)
        while iter_cnt != 0:
            y_pred = x @ w
            loss = y_pred-self.y
            gradient = (2/self.n)*(x.T@loss)
            iter_cnt-=1
            w -= a*gradient
        return w

    def stochastic_gradient_descent(self, a=0.0005, batch_size=2):
        x = self.add_bias(self.x)
        w = self.random_weights
        batch_sum = batch_size
        while batch_sum <= self.n:
            x_batch = x[batch_sum - batch_size: batch_sum]
            y_batch = self.y[batch_sum - batch_size: batch_sum]
            y_pred = x_batch @ w
            loss = y_pred - y_batch
            gradient = (2 / self.n) * (x_batch.T @ loss)
            batch_sum += batch_size
            w -= a * gradient
        return w

    def lasso_regression(self, a=0.0005, batch_size=2, l=1e-2):
        pass

    def ridge_regression(self, a=0.0005, batch_size=2, l=1e-2):
        x = self.add_bias(self.x)
        w = self.random_weights
        batch_sum = batch_size
        while batch_sum <= self.n:
            x_batch = x[batch_sum - batch_size: batch_sum]
            y_batch = self.y[batch_sum - batch_size: batch_sum]
            y_pred = (x_batch @ w) + l*(w**2)
            loss = y_pred - y_batch
            gradient = (2) * (x_batch.T @ loss) + 2*l*w
            batch_sum += batch_size
            w -= a * gradient
        return w

    def predict(self, w):
        return self.add_bias(self.x)@w

    #def calc_RSS(self):
    #    RSS = matrix_power((self.y-self.predict()), 2)
    #    return RSS


