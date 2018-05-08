from Classifiers.Classifier import classifier
from sklearn.tree import DecisionTreeClassifier


class decisionTreeClassifier(classifier):



    def __init__(self, model= DecisionTreeClassifier, criterion="gini", splitter="best", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                 random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                 class_weight=None,presort=False):
        """A decision tree classifier.
        """


        super(decisionTreeClassifier, self).__init__()
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.class_weight = class_weight
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.presort = presort
        self.model_name = "DT"
        self.model = model(self.criterion, self.splitter, self.max_depth, self.min_samples_split,
                           self.min_samples_leaf, self.min_weight_fraction_leaf, self.max_features,
                           self.random_state,self.max_leaf_nodes,self.min_impurity_decrease,
                           self.min_impurity_split,self.class_weight,self.presort)

    def fit(self, train_matrix, train_label, sample_weight=None):
        """Build a decision tree classifier from the training set (X, y).

       Parameters
       ----------
       train_matrix : array-like or sparse matrix, shape = [n_samples, n_features]
           The training input samples. Internally, it will be converted to
           ``dtype=np.float32`` and if a sparse matrix is provided
           to a sparse ``csc_matrix``.

       train_label : array-like, shape = [n_samples] or [n_samples, n_outputs]
           The target values (class labels) as integers or strings.

       sample_weight : array-like, shape = [n_samples] or None
           Sample weights. If None, then samples are equally weighted. Splits
           that would create child nodes with net zero or negative weight are
           ignored while searching for a split in each node. Splits are also
           ignored if they would result in any single class carrying a
           negative weight in either child node.

       """
        self.sample_weight = sample_weight
        return self.model.fit(train_matrix, train_label, self.sample_weight)

    def predict(self, test_matrix, check_input=True):
        """Predict class probabilities of the input samples test_matrix.

        The predicted class probability is the fraction of samples of the same
        class in a leaf.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Parameters
        ----------
        test_matrix: array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : bool
            Run check_array on X.

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        return self.model.predict(test_matrix, check_input)




    def get_alg(self):
        """Return the underlying model algorithm.
        Returns:
        algorithm used to train and test instances
        """

        return self.model_name





