import numpy as np

class SupervisedLearning():
    """

    """
    #print('''“All models are wrong, but some are useful.” — George Box''')
    def __init__(self, feature_vector, target_vector):
        self.x = feature_vector
        self.y = target_vector
        self.n = self.x.shape[0]
        self.m = self.x.shape[1]
        self.ones_array = np.ones((self.n, 1))

    def add_bias(self, x):
        x = np.concatenate((x, self.ones_array), axis=1)
        return x

    def train_test_split(self):
        pass

    def mse(self, y_pred):
        n = self.n
        se = ((self.y-y_pred)**2).sum()
        mse = se/n
        return mse

    def mae(self, y_pred):
        mae = (np.abs(self.y-y_pred).sum())/self.n
        return mae

    def mape(self, y_pred):
        mape = (np.abs((y_pred-self.y)/self.y).sum())/self.n
        return mape


