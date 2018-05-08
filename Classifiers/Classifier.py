import pickle

class classifier(object):

    """Abstract base class for a Classifer, which
    defines necessary operations for any classification
    model such as, training, prediction, classification
    probabilities, and decision function results.
    """

    def __init__(self):

        pass

    def fit(self, train_matrix, train_label, sample_weight):
         """Train on the set of training matrix and labels
         """
         raise NotImplementedError

    def predict(self, instances):
         """Predict classification labels for the set of instances.
         """
         raise NotImplementedError


    def get_alg(self):
        """Return the underlying model algorithm.
        Returns:
        algorithm used to train and test instances
        """
        raise NotImplementedError




