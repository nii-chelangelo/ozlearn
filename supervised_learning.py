import numpy as np

class SupervisedLearning():
    """
        The supervised learning process starts with gathering the data. The data for supervised
        learning is a collection of pairs (input, output). Input could be anything, for example, email
        messages, pictures, or sensor measurements. Outputs are usually real numbers, or labels (e.g.
        “spam”, “not_spam”, “cat”, “dog”, “mouse”, etc).

    """
    #print('''“All models are wrong, but some are useful.” — George Box''')
    def __init__(self, feature_vector, target_vector):
        self.x = feature_vector
        self.y = target_vector
        self.ones_array = np.ones((self.x.shape[0], 1))

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


