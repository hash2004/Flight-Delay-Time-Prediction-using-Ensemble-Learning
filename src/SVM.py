import numpy as np

class MultiKernelSVM:
    """
    A Support Vector Machine classifier with multiple kernel support implemented from scratch.
    This classifier allows the combination of multiple kernels (e.g., linear, polynomial, RBF)
    for training and prediction.
    """

    def __init__(self, C=1.0, kernels=['linear'], kernel_params=None, beta=None, max_iter=1000, tol=1e-3):
        """
        Initializes the MultiKernelSVM classifier.

        Parameters:
        - C: float, regularization parameter.
        - kernels: list of kernel names, e.g., ['linear', 'poly', 'rbf']
        - kernel_params: list of dictionaries containing parameters for each kernel.
        - beta: list of weights for each kernel.
        - max_iter: int, maximum number of iterations for the SMO algorithm.
        - tol: float, tolerance for stopping criterion.
        """
        self.C = C
        self.kernels = kernels
        self.kernel_params = kernel_params if kernel_params is not None else [{} for _ in kernels]
        self.beta = beta if beta is not None else [1.0 / len(kernels)] * len(kernels)
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = None
        self.b = 0
        self.X = None
        self.y = None

    def fit(self, X, y):
        """
        Fits the model using training data.

        Parameters:
        - X: numpy array of shape (n_samples, n_features)
        - y: numpy array of shape (n_samples,), labels {0, 1} or {-1, 1}
        """
        self.X = X
        # Convert labels to {-1, 1}
        y = np.where(y == 0, -1, y)
        y = np.where(y == 1, 1, y)
        self.y = y

        n_samples, n_features = X.shape

        # Compute combined kernel matrix
        K = self.compute_combined_kernel(X, X)

        # Initialize alpha and bias
        self.alpha = np.zeros(n_samples)
        self.b = 0

        # Implement SMO algorithm
        self.smo(K, y)

    def compute_combined_kernel(self, X1, X2):
        """
        Computes the combined kernel matrix.

        Parameters:
        - X1, X2: numpy arrays of shape (n_samples1, n_features), (n_samples2, n_features)

        Returns:
        - K_total: Combined kernel matrix of shape (n_samples1, n_samples2)
        """
        K_total = None
        for beta_m, kernel_name, params in zip(self.beta, self.kernels, self.kernel_params):
            K_m = self.compute_kernel(X1, X2, kernel_name, params)
            if K_total is None:
                K_total = beta_m * K_m
            else:
                K_total += beta_m * K_m
        return K_total

    def compute_kernel(self, X1, X2, kernel_name, params):
        """
        Computes the kernel matrix for a single kernel.

        Parameters:
        - X1, X2: numpy arrays
        - kernel_name: str, name of the kernel
        - params: dict, parameters for the kernel

        Returns:
        - K: Kernel matrix
        """
        if kernel_name == 'linear':
            return self.linear_kernel(X1, X2)
        elif kernel_name == 'poly':
            degree = params.get('degree', 3)
            coef0 = params.get('coef0', 1)
            gamma = params.get('gamma', 1)
            return self.polynomial_kernel(X1, X2, degree, gamma, coef0)
        elif kernel_name == 'rbf':
            gamma = params.get('gamma', None)
            if gamma is None:
                gamma = 1.0 / X1.shape[1]
            return self.rbf_kernel(X1, X2, gamma)
        else:
            raise ValueError(f"Unknown kernel '{kernel_name}'")

    def linear_kernel(self, X1, X2):
        return np.dot(X1, X2.T)

    def polynomial_kernel(self, X1, X2, degree, gamma, coef0):
        return (gamma * np.dot(X1, X2.T) + coef0) ** degree

    def rbf_kernel(self, X1, X2, gamma):
        sq_dists = np.sum(X1**2, axis=1).reshape(-1,1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return np.exp(-gamma * sq_dists)

    def smo(self, K, y):
        """
        Implements the Sequential Minimal Optimization algorithm.

        Parameters:
        - K: Kernel matrix
        - y: labels
        """
        n_samples = y.shape[0]
        alpha = self.alpha
        b = self.b

        passes = 0
        max_passes = 5
        iter_count = 0

        while passes < max_passes and iter_count < self.max_iter:
            num_changed_alphas = 0
            for i in range(n_samples):
                E_i = self.decision_function_single(K[i]) - y[i]
                if (y[i] * E_i < -self.tol and alpha[i] < self.C) or (y[i] * E_i > self.tol and alpha[i] > 0):
                    # Select j != i randomly
                    j = i
                    while j == i:
                        j = np.random.randint(0, n_samples)
                    E_j = self.decision_function_single(K[j]) - y[j]

                    alpha_i_old = alpha[i]
                    alpha_j_old = alpha[j]

                    # Compute L and H
                    if y[i] != y[j]:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = max(0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])

                    if L == H:
                        continue

                    # Compute eta
                    eta = 2 * K[i,j] - K[i,i] - K[j,j]
                    if eta >= 0:
                        continue

                    # Update alpha_j
                    alpha[j] = alpha_j_old - y[j] * (E_i - E_j) / eta

                    # Clip alpha_j
                    alpha[j] = np.clip(alpha[j], L, H)

                    if abs(alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    # Update alpha_i
                    alpha[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - alpha[j])

                    # Compute b1 and b2
                    b1 = b - E_i - y[i] * (alpha[i] - alpha_i_old) * K[i,i] - y[j] * (alpha[j] - alpha_j_old) * K[i,j]
                    b2 = b - E_j - y[i] * (alpha[i] - alpha_i_old) * K[i,j] - y[j] * (alpha[j] - alpha_j_old) * K[j,j]

                    # Update b
                    if 0 < alpha[i] < self.C:
                        b = b1
                    elif 0 < alpha[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0

                    num_changed_alphas +=1

            if num_changed_alphas == 0:
                passes +=1
            else:
                passes = 0

            iter_count +=1

        self.alpha = alpha
        self.b = b

    def decision_function_single(self, K_row):
        """
        Computes the decision function for a single sample.

        Parameters:
        - K_row: numpy array, kernel values between the sample and all training samples

        Returns:
        - decision_value: float
        """
        return np.sum(self.alpha * self.y * K_row) + self.b

    def decision_function(self, X):
        """
        Computes the decision function for given samples.

        Parameters:
        - X: numpy array, input samples

        Returns:
        - decision_values: numpy array, decision function values
        """
        K = self.compute_combined_kernel(X, self.X)
        return np.dot((self.alpha * self.y), K.T) + self.b

    def predict(self, X):
        """
        Predicts labels for given samples.

        Parameters:
        - X: numpy array, input samples

        Returns:
        - predictions: numpy array, predicted labels {0, 1}
        """
        decision_values = self.decision_function(X)
        return np.where(decision_values >= 0, 1, 0)

    def score(self, X, y):
        """
        Computes the accuracy of the model.

        Parameters:
        - X: numpy array, input samples
        - y: numpy array, true labels

        Returns:
        - accuracy: float, fraction of correctly predicted samples
        """
        y_pred = self.predict(X)
        y_true = np.where(y == -1, 0, y)
        return np.mean(y_pred == y_true)
