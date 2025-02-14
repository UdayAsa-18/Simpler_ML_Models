'''Note: To run all the models we can run the __init__.py'''
import numpy as np

class LDAModel:
    def __init__(self): 
        self.prior_probs = None #Defining prior probaility variable

    def _class_means_(self,X, y):
        means = [] #initiliazing the means variable as a list
        for c in np.unique(y):
            means.append(np.mean(X[y == c], axis=0)) #appending and evaluating means of every class using np.mean()
        return np.array(means)

    def _shared_class_cov(self,X, y):
        #initializing the covariance variable with zeros with dimensions of features of X (3072,3072)
        cov = np.zeros((X.shape[1], X.shape[1])) 

        #Evaluating shared covariance of all the classes 
        for c in np.unique(y):
            #evaluating difference of each sample of class with the mean of samples of each class
            diff = X[y == c] - self.class_means_[c]
            cov += self.prior_probs[c] * np.dot(diff.T, diff)
        return cov

    def fit(self, X, y):
        X_data = X.reshape(X.shape[0],-1) #reshaping X into 2D matrix
        self.class_means_ = self._class_means_(X_data, y) #claculating means for every class
        self.prior_probs = np.bincount(y) / len(y) #calculating prior probability of every class using bincount()
        self.shared_covariance = self._shared_class_cov(X_data, y) #calculating shared/pooled covariance for entire dataset
        self.weights_ = np.dot(self.class_means_, np.linalg.inv(self.shared_covariance)) #calculating weights
        self.intercept = -0.5 * np.diag(np.dot(np.dot(self.class_means_, np.linalg.inv(self.shared_covariance)), self.class_means_.T)) + np.log(self.prior_probs)#calculating intercept
        
        print(self.weights_.shape)
        print(self.intercept.shape)
    def predict(self,X):
        # TODO: Implement the predict method
        X_data = X.reshape(X.shape[0], -1)

        # Calculate discriminant scores
        _scores = np.dot(X_data, self.weights_.T) + self.intercept

        # Predict class labels
        predictions = np.argmax(_scores, axis=1)

        return predictions

class QDAModel:
    def __init__(self):
        self.class_priors_ = None #initializing the prior probability variability to hold probabilities of all classes

    def class_means_(self, X, y):
        means = [] #initiliazing the means variable as a list
        for c in np.unique(y):
            means.append(np.mean(X[y == c], axis=0)) #appending and evaluating means of every class using np.mean()
        return np.array(means)

    def class_cov(self, X, y):
        #initializing the covariance variable as an empty list
        covariances = []
        for c in np.unique(y):
            #evaluating difference of each sample of class with the mean of samples of each class
            diff = X[y == c] - self.class_means[c] 
            #evaluating covariance of each class
            cov = np.dot(diff.T, diff) / (len(X[y == c])-1)
            covariances.append(cov)
        return np.array(covariances)

    def fit(self, X, y):
        X_data = X.reshape(X.shape[0], -1) #reshaping X into 2D matrix 
        self.class_means = self.class_means_(X_data, y) #claculating means for every class
        self.class_priors_ = np.bincount(y) / len(y) #calculating prior probability of every class using bincount()
        self.class_covariances = self.class_cov(X_data, y) #calculating class covariance for each class


    def predict(self, X):
        X_data = X.reshape(X.shape[0], -1) #reshaping X into 2D matrix

        num_classes = len(self.class_priors_) #getting number classes using priors
        
        num_samples = X.shape[0] #assigning Total number of samples of X to num_samples

        # Initializing array to store discriminant function values for each class
        discriminant_values = np.zeros((num_samples, num_classes))
     
     #calculating discriminant values for every class 
        for c in range(num_classes):
            class_mean = self.class_means[c]
            prior = self.class_priors_[c]
             
            '''calculating the |det| of class covariance using slogdet() which returns a sign(-1,0,1) which tells
            whether the det value is positive or negative and it also gives log(det) which helps to handle
            large values so in discriminant function we just multiply sign with det_cov[log(det)] directly.
            '''
            sign,det_cov = np.linalg.slogdet(self.class_covariances[c])
            inv_cov = np.linalg.inv(self.class_covariances[c]) #calculating inverse covariance of each class

            # Calculating the constant term
            constant_term = -0.5 * sign * det_cov + np.log(prior)

            # Calculating the discriminant function for class c
            discriminant_values[:, c] = constant_term - 0.5 * np.sum(np.dot((X_data - class_mean), inv_cov) * (X_data - class_mean), axis=1)

        # Predicting the class with the highest discriminant function value
        predicted_classes = np.argmax(discriminant_values, axis=1)

        return predicted_classes
    
    
class GaussianNBModel:
    def __init__(self):
        #defining means,variances and class_priors variables
        self.class_means = None
        self.class_variances = None
        self.class_priors = None

    def fit(self, X, y):
        X_data = X.reshape(X.shape[0],-1) #reshaping X into a 2D matrix
        # Computing class priors
        unique_classes, class_counts = np.unique(y, return_counts=True)
        self.class_priors = class_counts / len(y)
        
        # Initializing arrays to store class means and variances
        num_features = X_data.shape[1]
        num_classes = len(unique_classes)
        self.class_means = np.zeros((num_classes, num_features))
        self.class_variances = np.zeros((num_classes, num_features))
        
        # Computing class means and variances
        for i, c in enumerate(unique_classes):
            X_c = X_data[y == c]
            self.class_means[i] = np.mean(X_c, axis=0)
            self.class_variances[i] = np.var(X_c, axis=0)

    def predict(self, X):
        X_data = X.reshape(X.shape[0],-1) #reshaping X into a 2D matrix
        # Computing the log-likelihood of each class for each sample
        log_likelihoods = np.zeros((X_data.shape[0], len(self.class_means)))
        for i, (mean, var) in enumerate(zip(self.class_means, self.class_variances)):
            log_likelihoods[:, i] = -0.5 * np.sum(np.log(2 * np.pi * var)) \
                                    - 0.5 * np.sum(((X_data - mean) ** 2) / var, axis=1)
        
        # Adding log-priors to log-likelihoods
        log_posteriors = log_likelihoods + np.log(self.class_priors)
        
        # Predicting the class with the highest posterior probability
        return np.argmax(log_posteriors, axis=1)



