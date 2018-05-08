from Classifiers.Classifier import classifier
from sklearn.svm import SVC


class svmClassifier(classifier):



    def __init__(self, model= SVC, C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                 decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
                 max_iter=-1, probability=False, random_state=None, shrinking=True,
                 tol=0.001, verbose=False):
        """C-Support Vector Classification.

        The implementation is based on libsvm. The fit time complexity
        is more than quadratic with the number of samples which makes it hard
        to scale to dataset with more than a couple of 10000 samples.

        The multiclass support is handled according to a one-vs-one scheme.
        """

        super(svmClassifier, self).__init__()
        self.C = C
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.coef0 = coef0
        self.decision_function_shape = decision_function_shape
        self.degree = degree
        self.gamma = gamma
        self.kernel = kernel
        self.max_iter = max_iter
        self.probability = probability
        self.random_state = random_state
        self.shrinking = shrinking
        self.tol = tol
        self.verbose = verbose
        self.model_name = "SVM"
        self.model = model(self.C, self.kernel, self.degree, self.gamma, self.coef0, self.probability,
                           self.shrinking, self.tol, self.cache_size, self.class_weight, self.verbose,
                           self.max_iter, self.decision_function_shape, self.random_state)

    def fit(self, train_matrix, train_label, sample_weight=None ):
        """Fit the model according to the given training data.

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

