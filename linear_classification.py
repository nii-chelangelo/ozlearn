import numpy as np
from numpy.linalg import inv, matrix_power

from supervised_learning import SupervisedLearning
class LinearClassification(SupervisedLearning):
    """
        Это строка документирования
        x: feature vector
        Y: target vector
    """
    def __init__(self, feature_vector, target_vector):
        super().__init__(feature_vector, target_vector)
        self.x = feature_vector.values
        self.y = target_vector.values

    def svm(self):
        x = []
        w = []
        margin = x @ w # misclassification loss
        pass

    def logistic_regression(self, a=0.1, iter_cnt=200):
        w = np.ones((self.m, 1))
        x = self.x
        b = 0
        while iter_cnt != 0:
            z = (x@w) + b
            p = 1/(1+np.exp(-z))
            gradient = (1/self.n)*(x.T @ (p - self.y))
            iter_cnt -= 1
            w -= a * gradient
            b -= a*np.sum((1/self.n)*(x.T @ (p - self.y)))
        return w, b


    def predict(self, w, b):
        p = 1/(1+np.exp(-(self.x @ w + b)))
        y_pred = np.where(p > 0.5, 1, 0)
        return y_pred

