import numpy as np

class DecisionTreeClassifier:
    """
    A Decision Tree Classifier implemented from scratch.
    Supports both Gini impurity and entropy criteria for splitting,
    handles numerical and categorical features, and supports multi-class classification.
    """
    
    class Node:
        """
        Represents a node in the decision tree.
        """
        def __init__(self, gini, entropy, num_samples, num_samples_per_class, predicted_class):
            self.gini = gini
            self.entropy = entropy
            self.num_samples = num_samples
            self.num_samples_per_class = num_samples_per_class
            self.predicted_class = predicted_class
            self.feature_index = None
            self.threshold = None
            self.left = None
            self.right = None
    
    def __init__(self,
                 criterion='gini',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features=None,
                 random_state=None):
        """
        Initializes the Decision Tree Classifier.

        Parameters:
        - criterion: str, 'gini' for Gini impurity or 'entropy' for information gain
        - max_depth: int, maximum depth of the tree
        - min_samples_split: int, minimum number of samples required to split an internal node
        - min_samples_leaf: int, minimum number of samples required to be at a leaf node
        - max_features: int or None, number of features to consider when looking for the best split
        - random_state: int or None, seed for random number generator
        """
        self.criterion = criterion
        if self.criterion not in ('gini', 'entropy'):
            raise ValueError("Criterion must be 'gini' or 'entropy'")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_classes_ = None
        self.n_features_ = None
        self.tree_ = None
    
    def fit(self, X, y):
        """
        Build a decision tree classifier from the training set (X, y).

        Parameters:
        - X: numpy array of shape (n_samples, n_features)
        - y: numpy array of shape (n_samples,), target class labels
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        
        if self.max_features is None:
            self.max_features = self.n_features_
        elif isinstance(self.max_features, int):
            self.max_features = min(self.max_features, self.n_features_)
        else:
            raise ValueError("max_features should be int or None")
        
        self.tree_ = self._grow_tree(X, y)
    
    def predict(self, X):
        """
        Predict class for X.

        Parameters:
        - X: numpy array of shape (n_samples, n_features)

        Returns:
        - y_pred: numpy array of shape (n_samples,), predicted class labels
        """
        return np.array([self._predict(inputs) for inputs in X])
    
    def _gini(self, y):
        """
        Compute Gini impurity for labels y.
        """
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in set(y))
    
    def _entropy(self, y):
        """
        Compute entropy for labels y.
        """
        m = len(y)
        return -sum((np.sum(y == c) / m) * np.log2(np.sum(y == c) / m + 1e-9) for c in set(y))
    
    def _best_split(self, X, y):
        """
        Find the best split for a node.

        Returns:
        - best_idx: Index of the feature to split on
        - best_thr: Threshold value to split at
        """
        m, n = X.shape
        if m <= 1:
            return None, None
        
        # Initialize parameters
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        if self.criterion == 'gini':
            best_criterion = self._gini(y)
        else:
            best_criterion = self._entropy(y)
        best_idx, best_thr = None, None
        
        features = np.random.choice(n, self.max_features, replace=False)
        
        for idx in features:
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            
            for i in range(1, m):  # possible split positions
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                
                criterion_left = self._gini(classes[:i]) if self.criterion == 'gini' else self._entropy(classes[:i])
                criterion_right = self._gini(classes[i:]) if self.criterion == 'gini' else self._entropy(classes[i:])
                
                # Calculate the weighted criterion
                criterion = (i * criterion_left + (m - i) * criterion_right) / m
                
                # Update best split if needed
                if thresholds[i] == thresholds[i - 1]:
                    continue
                
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2  # midpoint

        return best_idx, best_thr
    
    def _grow_tree(self, X, y, depth=0):
        """
        Recursively build the tree.
        """
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = self.Node(
            gini=self._gini(y),
            entropy=self._entropy(y),
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )
        
        # Check stopping conditions
        if depth < self.max_depth if self.max_depth is not None else True:
            if node.num_samples >= self.min_samples_split and node.gini > 0:
                idx, thr = self._best_split(X, y)
                if idx is not None:
                    indices_left = X[:, idx] < thr
                    X_left, y_left = X[indices_left], y[indices_left]
                    X_right, y_right = X[~indices_left], y[~indices_left]
                    
                    if y_left.size >= self.min_samples_leaf and y_right.size >= self.min_samples_leaf:
                        node.feature_index = idx
                        node.threshold = thr
                        node.left = self._grow_tree(X_left, y_left, depth + 1)
                        node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node
    
    def _predict(self, inputs):
        """
        Predict class for a single sample.
        """
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class
