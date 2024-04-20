import numpy as np
import heapq

class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind = "breed_identifying"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind =task_kind
        self.training_data = None
        self.training_labels = None

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N, D)
                training_labels (np.array): labels of shape (N,) or (N, 2)
            Returns:
                pred_labels (np.array): labels of shape (N,) or (N, 2)
        """
        self.training_data = training_data
        self.training_labels = training_labels
        return self.predict(self.training_data)

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N, D)
            Returns:
                test_labels (np.array): labels of shape (N,) or (N, 2)
        """
        # TODO Vectorize predictors.
        def predict_classification(neighbours_labels):
            return np.array([
                np.argmax(np.bincount(neighbours_labels[i]))
                for i in range(test_data.shape[0])
            ])
        
        def predict_regression(neighbours_labels):
            return np.mean(neighbours_labels, axis=1)
        
        predict = \
            predict_classification if self.task_kind == "breed_identifying" \
            else predict_regression
        
        test_labels = \
            np.zeros(test_data.shape[0]) if self.task_kind == "breed_identifying" \
            else np.zeros((test_data.shape[0], self.training_labels.shape[1]))
            
        distances = np.linalg.norm(
            np.expand_dims(self.training_data, 0) \
            - np.expand_dims(test_data, 1),
            axis=2
        )
        
        neighbours_indexes = np.argpartition(distances, self.k, axis=1)[:, :self.k]
        neighbours_labels = self.training_labels[neighbours_indexes]
        test_labels = predict(neighbours_labels)

        # for test_index in range(test_data.shape[0]):
        #     test_labels[test_index] = predict(neighbours_labels[test_index])
            
        return test_labels
    