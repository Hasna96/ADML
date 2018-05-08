from Classifiers.Classifier import classifier
from sklearn.naive_bayes import MultinomialNB


class multiNomialNaiveBayesClassifier(classifier):

    def __init__(self, model_name= MultinomialNB, alpha =1.0, fit_prior=True, class_prior=None):
        """Naive Bayes classifier for multinomial models

        The multinomial Naive Bayes classifier is suitable for classification with
        discrete features (e.g., word counts for text classification). The
        multinomial distribution normally requires integer feature counts. However,
        in practice, fractional counts such as tf-idf may also work.
        """

        super(multiNomialNaiveBayesClassifier, self).__init__()
        self.alpha = alpha
        self.fit_prior =fit_prior
        self.class_prior = class_prior
        self.model = model_name(self.alpha, self.fit_prior, self.class_prior )
        self.model_name = "MultinomialNB"

    def fit(self, train_matrix, train_label, sample_weight=None ):
        """Fit Naive Bayes classifier according to train_matrix, train_label

        Parameters
        ----------
        train_matrix : {array-like, sparse matrix}, shape = [n_samples, n_features]
           Training vectors, where n_samples is the number of samples and
           n_features is the number of features.

        train_label: array-like, shape = [n_samples]
           Target values.

        sample_weight : array-like, shape = [n_samples], (default=None)
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
           Returns self.
        """
        self.sample_weight = sample_weight

        return self.model.fit(train_matrix, train_label, self.sample_weight)

    def predict(self, test_matrix):
        """Perform classification on an array of test vectors test_matrix.

        Parameters
        ----------
        test_matrix : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples]
            Predicted target values for test_matrix
        """
        return self.model.predict(test_matrix)



    def get_alg(self):
        """Return the underlying model algorithm.
        Returns:
        algorithm used to train and test instances
        """

        return self.model_name








