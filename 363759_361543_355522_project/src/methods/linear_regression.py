import numpy as np
import sys

class LinearRegression(object):
    """
        Linear regressor object. 
        Note: This class will implement BOTH linear regression and ridge regression.
        Recall that linear regression is just ridge regression with lambda=0.
    """

    def __init__(self, lmda):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        self.lmda = lmda
        self.N = None
        self.D = None # D already counts the bias term
        self.C = None
        self.weights = None

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,C)
            Returns:
                pred_labels (np.array): target of shape (N,C)
        """
        ##
        ###
        #### YOUR CODE HERE!
        ###
        ##
        self.N = training_data.shape[0]
        self.D = training_data.reshape((self.N, -1)).shape[-1]
        self.C = training_labels.reshape((self.N, -1)).shape[-1]
        I = np.eye(self.D)
        inverse = np.linalg.inv(training_data.T @ training_data + I * self.lmda)
        self.weights = inverse @ training_data.T @ training_labels
        
        return self.predict(training_data)


    def predict(self, test_data):
        """
            Runs prediction on the test data.
            
            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,C)
        """
        ##
        ###
        #### YOUR CODE HERE!
        ###
        ##
        return test_data @ self.weights
