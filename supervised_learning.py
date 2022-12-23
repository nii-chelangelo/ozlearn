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
        self.random_weights = np.random.default_rng(37).normal(0.5, 0.1, (self.m+1, 1)) # плюс 1 для bais
        self.ones_array = np.ones((self.n, 1))

    def add_bias(self, x):
        x = np.concatenate((x, self.ones_array), axis=1)
        return x

    def train_test_split(self):
        pass

    def mse(self, y_pred):
        n = self.x.shape[0]
        se = ((self.y-y_pred)**2).sum()
        mse = se/n
        return mse


