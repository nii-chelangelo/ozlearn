class SupervisedLearning():
    """
        The supervised learning process starts with gathering the data. The data for supervised
        learning is a collection of pairs (input, output). Input could be anything, for example, email
        messages, pictures, or sensor measurements. Outputs are usually real numbers, or labels (e.g.
        “spam”, “not_spam”, “cat”, “dog”, “mouse”, etc).

    """
    print('''“All models are wrong, but some are useful.” — George Box''')
    def __init__(self, feature_vector, target_vector):
        self.x = feature_vector
        self.y = target_vector

    def train_test_split(self):
        pass
