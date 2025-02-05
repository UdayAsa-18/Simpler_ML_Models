import pickle
import numpy as np

class LinearRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Linear Regression using Gradient Descent.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None
        self.batch_loss_history = []

    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Fit a linear model.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_loss_history = []
        # TODO: Initialize the weights and bias based on the shape of X and y.
        self.weights = np.random.randn(X.shape[1])
        self.bias = (np.random.rand() * 2) - 1
    
        data_val = int(0.1 * X.shape[0])
        data_train = X.shape[0] - data_val

        X_train, y_train = X[:data_train], y[:data_train]
        X_val, y_val = X[data_train:], y[data_train:]

        best_weights_term = self.weights.copy()
        best_bias_term = self.bias

        best_val_loss = float('inf')
        count_patience = 0

        # TODO: Implement the training loop.
        for epoch in range(max_epochs):
               indices = np.arange(data_train)
               np.random.shuffle(indices)
               X_train, y_train = X_train[indices], y_train[indices]

               for i in range(0, data_train, batch_size):
                    end_index = i+batch_size if (i+batch_size)<= data_train else data_train
                    batch_train_X, batch_train_y = X_train[:end_index], y_train[:end_index]

                    y_hat = np.dot(batch_train_X,self.weights) + self.bias # y = mx+c form 

                    mse_loss = np.mean((y_hat - batch_train_y)**2)

                    #adding regularization term 
                
                    regularization_var = 0.5*regularization*np.sum((self.weights)**2)

                    total_loss = mse_loss +regularization_var

                    gradient_weights = (np.dot(batch_train_X.T, (y_hat - batch_train_y))/batch_size) + regularization*self.weights

                    gradient_bias = np.sum(y_hat - batch_train_y)/batch_size

                    learning_rate = 0.01 #define learning rate

                    self.weights -= learning_rate*gradient_weights # new_weight  = old_weight - step size where step size = slope*learning rate
                    self.bias -= learning_rate*gradient_bias # new_intercept = old_intercept - step size
              
               self.batch_loss_history.append(total_loss/self.batch_size)

               y_pred_val = np.dot(X_val,self.weights) + self.bias
               val_loss = np.mean((y_pred_val-y_val)**2)

               if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights_term = self.weights.copy()
                    best_bias_term = self.bias
                    count_patience = 0
               else:
                    count_patience+=1
                    
               if count_patience >= patience:
                    print(f"Early stopping has been reached after {epoch+1} epochs")
                    break
               
        self.weights = best_weights_term
        self.bias = best_bias_term
        
               
    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        """
        # TODO: Implement the prediction function.

        if self.weights is None or self.bias is None:
             raise ValueError("Model has not been trained. Please invoke the fit() method.")
        y_hat_pred = np.dot(X,self.weights) + self.bias

        return y_hat_pred

    def score(self, X, y):
        """Evaluate the linear model using the mean squared error.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        """
        # TODO: Implement the scoring function.
        if self.weights is None or self.bias is None:
             raise ValueError("Model has not been trained. Please invoke the fit() method.")
        
        y_hat = self.predict(X)

        mse = np.mean((y_hat - y)**2)
        
        return mse
    
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
