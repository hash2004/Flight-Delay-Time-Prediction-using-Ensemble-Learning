import numpy as np

class LogisticRegressionClassifier:
    """
    A Logistic Regression Classifier implemented from scratch.
    Supports binary classification, uses gradient descent or stochastic gradient descent for optimization,
    and includes options for regularization (L1 and L2 penalties).
    """
    
    def __init__(self, 
                 learning_rate=0.01,
                 num_iterations=1000,
                 fit_intercept=True,
                 verbose=False,
                 method='gradient_descent',
                 batch_size=32,
                 regularization=None,
                 reg_strength=0.01,
                 random_state=None):
        """
        Initializes the Logistic Regression Classifier.

        Parameters:
        - learning_rate: float, step size for gradient descent
        - num_iterations: int, number of iterations for the optimization algorithm
        - fit_intercept: bool, whether to include an intercept term
        - verbose: bool, whether to print progress messages
        - method: str, 'gradient_descent' or 'stochastic_gradient_descent'
        - batch_size: int, size of the mini-batches for SGD
        - regularization: str, 'l1', 'l2', or None for no regularization
        - reg_strength: float, regularization strength
        - random_state: int or None, seed for random number generator
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.method = method
        self.batch_size = batch_size
        self.regularization = regularization
        self.reg_strength = reg_strength
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = None
    
    def _initialize_weights(self, n_features):
        """
        Initializes the weights (coefficients) of the model.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.coef_ = np.zeros(n_features)
        if self.fit_intercept:
            self.intercept_ = 0.0
    
    def _add_intercept(self, X):
        """
        Adds an intercept term to the data matrix X.
        """
        intercept = np.ones((X.shape[0], 1))
        return np.hstack((intercept, X))
    
    def _sigmoid(self, z):
        """
        Computes the sigmoid function.
        """
        return 1 / (1 + np.exp(-z))
    
    def _loss(self, h, y):
        """
        Computes the logistic loss function.
        """
        epsilon = 1e-15  # to prevent log(0)
        loss = -y * np.log(h + epsilon) - (1 - y) * np.log(1 - h + epsilon)
        return np.mean(loss)
    
    def _compute_gradient(self, X, y, h):
        """
        Computes the gradient of the loss function with respect to weights.
        """
        m = y.size
        gradient = np.dot(X.T, (h - y)) / m
        
        if self.regularization == 'l2':
            gradient += (self.reg_strength / m) * self.coef_
        elif self.regularization == 'l1':
            gradient += (self.reg_strength / m) * np.sign(self.coef_)
        
        return gradient
    
    def fit(self, X, y):
        """
        Fits the model according to the given training data.

        Parameters:
        - X: numpy array of shape (n_samples, n_features)
        - y: numpy array of shape (n_samples,), target binary class labels (0 or 1)
        """
        # Initialize weights
        if self.fit_intercept:
            X = self._add_intercept(X)
        n_samples, n_features = X.shape
        self._initialize_weights(n_features)
        
        # Gradient Descent optimization
        if self.method == 'gradient_descent':
            for i in range(self.num_iterations):
                z = np.dot(X, self.coef_)
                h = self._sigmoid(z)
                gradient = self._compute_gradient(X, y, h)
                
                # Update weights
                self.coef_ -= self.learning_rate * gradient
                
                # Verbose output
                if self.verbose and i % 100 == 0:
                    loss = self._loss(h, y)
                    print(f'Iteration {i}: loss = {loss}')
        
        # Stochastic Gradient Descent optimization
        elif self.method == 'stochastic_gradient_descent':
            if self.random_state is not None:
                np.random.seed(self.random_state)
            for i in range(self.num_iterations):
                indices = np.arange(n_samples)
                np.random.shuffle(indices)
                for start_idx in range(0, n_samples, self.batch_size):
                    batch_indices = indices[start_idx:start_idx + self.batch_size]
                    X_batch = X[batch_indices]
                    y_batch = y[batch_indices]
                    
                    z = np.dot(X_batch, self.coef_)
                    h = self._sigmoid(z)
                    gradient = self._compute_gradient(X_batch, y_batch, h)
                    
                    # Update weights
                    self.coef_ -= self.learning_rate * gradient
                
                # Verbose output
                if self.verbose and i % 100 == 0:
                    z_full = np.dot(X, self.coef_)
                    h_full = self._sigmoid(z_full)
                    loss = self._loss(h_full, y)
                    print(f'Iteration {i}: loss = {loss}')
        
        else:
            raise ValueError("Method must be 'gradient_descent' or 'stochastic_gradient_descent'")
        
        # Separate intercept from coefficients if fit_intercept is True
        if self.fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]
        else:
            self.intercept_ = 0.0
    
    def predict_proba(self, X):
        """
        Predicts probabilities for each class label.

        Parameters:
        - X: numpy array of shape (n_samples, n_features)

        Returns:
        - probabilities: numpy array of shape (n_samples,), probabilities of the positive class
        """
        if self.fit_intercept:
            X = self._add_intercept(X)
        z = np.dot(X, np.hstack(([self.intercept_], self.coef_)))
        return self._sigmoid(z)
    
    def predict(self, X):
        """
        Predicts class labels for samples in X.

        Parameters:
        - X: numpy array of shape (n_samples, n_features)

        Returns:
        - y_pred: numpy array of shape (n_samples,), predicted class labels (0 or 1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
    
    def score(self, X, y):
        """
        Computes the accuracy of the model.

        Parameters:
        - X: numpy array of shape (n_samples, n_features)
        - y: numpy array of shape (n_samples,), true class labels

        Returns:
        - accuracy: float, fraction of correctly predicted samples
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
