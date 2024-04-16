import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters


    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        self.N = training_data.shape[0] # number of training instances
        self.D = training_data.shape[1] # number of features
        self.C = get_n_classes(training_labels) # number of classes

        onehot_training_labels = label_to_onehot(training_labels)

        # Random initializantion
        self.weights = np.random.normal(0, 1, (self.D, self.C)) ## Initialization in (0, 1)

        # Iterate, updating the weights using the gradient, 
        # until self.max_iters is reached
        for _ in range(self.max_iters):
            
            gradient = self.compute_gradient(training_data, onehot_training_labels)
            self.weights -= self.lr * gradient

        #Use the computed weights to predict the labels for training_data
        pred_labels = self.predict(training_data)
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """

        # Use softmax to compute the probability that each instance
        # belongs to each class and return the one the class corresponding
        # to the maximum probabilty
        probabilities = self.softmax(test_data)
        pred_labels = np.argmax(probabilities, axis = 1)
        print(pred_labels)

        return pred_labels

    def softmax(self, data):
        """
        Computes the probability matrix (the probability
        that each instance belongs to each class).

        Arguments:
            data (array): input data of shape (N,D)
        Returns:
            prob (array): the probability matrix of shape (N, C)
        """
        exp_dot_prod = np.exp(data @ self.weights)
        exp_dot_prod_sum = np.sum(exp_dot_prod, axis = 0)
        prob = exp_dot_prod / exp_dot_prod_sum
        return prob
    
    def compute_gradient(self, data, labels):
        """
        Computes the gradient of the logistic regression model.

        Arguments:
            data (array): training data of shape (N,D)
            labels (array): the actual labels of shape (N, )
            
        Returns:
            gradient (array): the gradient of shape (D, C)
        """
        gradient = data.T @ (self.softmax(data) - labels)
        return gradient


