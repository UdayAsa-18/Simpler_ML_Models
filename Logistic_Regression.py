import pickle
import numpy as np
class LogisticRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Subtracting max to avoid numerical instability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _compute_loss(self, X_batch, y_batch):
        z = np.dot(X_batch, self.weights) + self.bias
        probabilities = self._softmax(z)
        epsilon = 1e-5  # Small value to avoid division by zero in logarithm
        # Cross-entropy loss function
        loss = -np.mean(np.sum(y_batch * np.log(probabilities + epsilon), axis=1))
        return loss

    def _one_hot_encode(self, y):
        y_encoded = np.zeros((len(y), self.num_classes))
        for i, label in enumerate(y):
            y_encoded[i, label] = 1
        return y_encoded

    def fit(self, X, y, batch_size=32, max_epochs=1000, regularization=0, learning_rate=0):
        self.num_classes = len(np.unique(y))
        self.num_features = X.shape[1] if len(X.shape) > 1 else 1
        self.weights = np.random.randn(self.num_features, self.num_classes)  # Initialize weights

        self.bias = np.zeros(self.num_classes)  # Initialize bias
        self.batch_size = batch_size
        self.max_epochs = max_epochs

        data_val = int(0.1 * X.shape[0])
        data_train = X.shape[0] - data_val

        
        X_train, X_val = X[:data_train], X[data_train:]
        y_train, y_val = y[:data_train], y[data_train:]

        best_weights = self.weights.copy()
        best_bias = self.bias.copy()
        best_val_loss = float('inf')

        for epoch in range(max_epochs):
            indices = np.arange(data_train)
            np.random.shuffle(indices)
            X_train, y_train = X_train[indices], y_train[indices]

            for i in range(0, data_train, batch_size):
                end_index = i+batch_size if (i+batch_size)<= data_train else data_train
                X_batch, y_batch = X_train[:end_index], y_train[:end_index]


                z = np.dot(X_batch, self.weights) + self.bias
                probabilities = self._softmax(z)

                # Computing gradients
                errors = probabilities - self._one_hot_encode(y_batch)
                gradients = np.dot(X_batch.T, errors) / len(X_batch)
                bias_gradient = np.mean(errors, axis=0)

                # Regularization
                gradients += regularization * self.weights

                # Updating weights and bias
                self.weights -= learning_rate * gradients
                self.bias -= learning_rate * bias_gradient

            # # Computing validation loss
            val_loss = self._compute_loss(X_val, self._one_hot_encode(y_val))

            # Early stopping
            if val_loss < best_val_loss:
                best_weights = self.weights.copy()
                best_bias = self.bias.copy()
                best_val_loss = val_loss
                patience_count = 0
            # else:
            #     patience_count += 1
            #     if patience_count >= patience:
            #         print(f"Early stopping after {epoch+1} epochs")
            #         break

        # Updating weights and bias with the best values
        self.weights = best_weights
        self.bias = best_bias

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        probabilities = self._softmax(z)
        return np.argmax(probabilities, axis=1)
    
    def score(self, X, y):
        predictions = self.predict(X)
        correct_predictions = np.sum(predictions == y)
        total_samples = len(y)
        accuracy = correct_predictions / total_samples
        return accuracy
    
    def save(self, filepath):
        """Save the model parameters to a file.

        Parameters:
        -----------
        filepath: str
            The file path where the model parameters will be saved.
        """
        with open(filepath, 'wb') as f:
            pickle.dump({"weights": self.weights, "bias": self.bias}, f)

    def load(self, filepath):
        """Load the model parameters from a file.

        Parameters:
        -----------
        filepath: str
            The file path from where the model parameters will be loaded.
        """
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
            self.weights = params["weights"]
            self.bias = params["bias"]

        model_params = {"weights":self.weights, "bias":self.bias}

        return model_params
