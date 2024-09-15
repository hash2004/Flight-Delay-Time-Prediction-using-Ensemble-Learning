import numpy as np

class MLPClassifier:
    """
    A Multi-Layer Perceptron (MLP) Classifier implemented from scratch.
    Supports multiple hidden layers, various activation functions, backpropagation,
    L2 regularization, and dropout for regularization.
    """
    
    def __init__(self, 
                 hidden_layer_sizes=(100,), 
                 activation='relu', 
                 learning_rate=0.01, 
                 num_iterations=1000, 
                 batch_size=32, 
                 dropout=0.0,
                 regularization='l2', 
                 reg_strength=0.01, 
                 verbose=False, 
                 random_state=None):
        """
        Initializes the MLP Classifier.

        Parameters:
        - hidden_layer_sizes: tuple, number of neurons in each hidden layer
        - activation: str, 'relu' or 'sigmoid' for activation functions
        - learning_rate: float, learning rate for gradient descent
        - num_iterations: int, number of iterations for optimization
        - batch_size: int, size of mini-batches for SGD
        - dropout: float, dropout rate (between 0 and 1) for regularization
        - regularization: str, 'l2' or None for no regularization
        - reg_strength: float, strength of L2 regularization
        - verbose: bool, whether to print progress during training
        - random_state: int or None, seed for random number generator
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.dropout = dropout
        self.regularization = regularization
        self.reg_strength = reg_strength
        self.verbose = verbose
        self.random_state = random_state
        self.weights = []
        self.biases = []
    
    def _initialize_weights(self, layer_dims):
        """
        Initializes the weights and biases for the network.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.weights = []
        self.biases = []
        
        for i in range(1, len(layer_dims)):
            weight = np.random.randn(layer_dims[i], layer_dims[i - 1]) * 0.01
            bias = np.zeros((layer_dims[i], 1))
            self.weights.append(weight)
            self.biases.append(bias)
    
    def _activation(self, Z):
        """
        Computes the activation for forward propagation.
        """
        if self.activation == 'relu':
            return np.maximum(0, Z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        else:
            raise ValueError("Activation must be 'relu' or 'sigmoid'")
    
    def _activation_derivative(self, Z):
        """
        Computes the derivative of the activation function.
        """
        if self.activation == 'relu':
            return (Z > 0).astype(float)
        elif self.activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-Z))
            return sig * (1 - sig)
    
    def _forward_propagation(self, X):
        """
        Implements the forward propagation step.
        """
        A = X
        cache = {'A0': X}
        
        for i in range(len(self.weights) - 1):
            Z = np.dot(self.weights[i], A) + self.biases[i]
            A = self._activation(Z)
            if self.dropout > 0:
                dropout_mask = (np.random.rand(*A.shape) < (1 - self.dropout)) / (1 - self.dropout)
                A = A * dropout_mask
                cache[f'D{i+1}'] = dropout_mask
            cache[f'Z{i+1}'] = Z
            cache[f'A{i+1}'] = A
        
        # Output layer (no activation for output)
        ZL = np.dot(self.weights[-1], A) + self.biases[-1]
        AL = self._sigmoid(ZL)  # Output layer uses sigmoid activation
        cache[f'Z{len(self.weights)}'] = ZL
        cache[f'A{len(self.weights)}'] = AL
        
        return AL, cache
    
    def _compute_loss(self, AL, Y):
        """
        Computes the cross-entropy loss with optional L2 regularization.
        """
        m = Y.shape[1]
        epsilon = 1e-15  # To prevent log(0)
        loss = -np.sum(Y * np.log(AL + epsilon) + (1 - Y) * np.log(1 - AL + epsilon)) / m
        
        if self.regularization == 'l2':
            reg_loss = (self.reg_strength / (2 * m)) * sum(np.sum(np.square(W)) for W in self.weights)
            loss += reg_loss
        
        return loss
    
    def _backward_propagation(self, AL, Y, cache):
        """
        Implements the backward propagation step.
        """
        m = Y.shape[1]
        grads = {}
        
        # Output layer gradient
        dZL = AL - Y
        grads[f'dW{len(self.weights)}'] = np.dot(dZL, cache[f'A{len(self.weights) - 1}'].T) / m
        grads[f'db{len(self.weights)}'] = np.sum(dZL, axis=1, keepdims=True) / m
        
        if self.regularization == 'l2':
            grads[f'dW{len(self.weights)}'] += (self.reg_strength / m) * self.weights[-1]
        
        # Backpropagate through the hidden layers
        for i in reversed(range(len(self.weights) - 1)):
            dA = np.dot(self.weights[i + 1].T, dZL)
            if self.dropout > 0:
                dA = dA * cache[f'D{i+1}']
            dZ = dA * self._activation_derivative(cache[f'Z{i+1}'])
            grads[f'dW{i+1}'] = np.dot(dZ, cache[f'A{i}'].T) / m
            grads[f'db{i+1}'] = np.sum(dZ, axis=1, keepdims=True) / m
            dZL = dZ
            
            if self.regularization == 'l2':
                grads[f'dW{i+1}'] += (self.reg_strength / m) * self.weights[i]
        
        return grads
    
    def _update_parameters(self, grads):
        """
        Updates the weights and biases using gradient descent.
        """
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads[f'dW{i+1}']
            self.biases[i] -= self.learning_rate * grads[f'db{i+1}']
    
    def fit(self, X, y):
        """
        Trains the MLP model using mini-batch gradient descent.

        Parameters:
        - X: numpy array of shape (n_samples, n_features), training data
        - y: numpy array of shape (n_samples,), target binary class labels (0 or 1)
        """
        X = X.T  # Transpose to shape (n_features, n_samples)
        Y = y.reshape(1, -1)  # Shape (1, n_samples)
        n_samples, n_features = X.shape[1], X.shape[0]
        
        # Initialize layer dimensions
        layer_dims = [n_features] + list(self.hidden_layer_sizes) + [1]
        
        # Initialize weights and biases
        self._initialize_weights(layer_dims)
        
        for i in range(self.num_iterations):
            AL, cache = self._forward_propagation(X)
            loss = self._compute_loss(AL, Y)
            grads = self._backward_propagation(AL, Y, cache)
            self._update_parameters(grads)
            
            if self.verbose and i % 100 == 0:
                print(f'Iteration {i}: Loss = {loss}')
    
    def predict_proba(self, X):
        """
        Predicts probabilities for each class label.

        Parameters:
        - X: numpy array of shape (n_samples, n_features)

        Returns:
        - probabilities: numpy array of shape (n_samples,), probabilities of the positive class
        """
        X = X.T  # Transpose to shape (n_features, n_samples)
        AL, _ = self._forward_propagation(X)
        return AL.flatten()
    
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
    
    def _sigmoid(self, Z):
        """
        Sigmoid activation function (for the output layer).
        """
        return 1 / (1 + np.exp(-Z))
