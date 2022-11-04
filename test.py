#!/usr/bin/env python
# coding: utf-8

# ### Assignment 1

# In[ ]:


import sys

import os

print("Current working directory: ", os.getcwd())

# In[ ]:


# Import the necessary libraries/packages here
import numpy as np

# import math


# ### Helpful Notes:
# 1. Dataset 1: a linearly separable dataset where you can test the correctness of your base learner and boosting algorithms
#    
#    300 samples 2 features
#    
#    ![dataset1.png](./dataset1.png)
#    
#    Generally speaking, your learners shall 100% correctly classify the data in dataset 1.
# 
# 2. Dataset 2 ~ 4 : non-linearly separable cases, applying descent boosting techniques can be beneficial
#    
#    Dataset 2: 300 samples 2 features. In comparison to the performance of your single base learner, does your boosting algorithm perferm better?
#       
#    ![dataset2.png](./dataset2.png)
#       
#    Dataset 3: 400 samples 2 features (challenging)
# 
#       A good classifier shall obtain a ellipse-like decision boundary on this dataset. Can your algorithms handle this dataset? If not, can you try to give reasonable explanations?
# 
#    ![dataset3.png](./dataset3.png)
# 
#    Dataset 4: 3000 samples 10 features (more challenging)
#    
#       This is more or less the higher dimensional version of dataset3. We visualize the first two features of dataset 3, As it is shown in the following figure, they are non-linearly separable. 
#       
#       How do your algorithms perform?
# 
#    ![dataset4.png](./dataset4.png)
# 
#    
# 3. The data is also provided in csv format:
#    1. Feature columns and a label column 
#    
# HINTs: 
# 1. Split the data into two parts (i.e., training data and test data).
# 2. Draw decision boundary (surface) of your classifiers (on dataset 1 & 2) can be helpful.
# 3. Carefully design your experiments so that you can understand the influence of increasing or decreasing some parameters (e.g., learning rate, number of base learners in boosting Alg.)
# 4. Make smart implementations (e.g., vectorization using numpy to avoid some nested-loops in python), so that you can efficiently run more experiments
# 5. The performance of your classifiers is not of high priority in this assignment.
#    1. The datasets are all artificially generated (toy) data, in principle, there is no need to preprocess the data.
#    2. Constructive discussions on your findings are more important. If the results are not good, try to find out the reasons.
#    3. We hope this assignment can help you fill in the gap between theory and application.
# 6. You are encouraged to implement not only Adaboost but also other boosting algorithms of your choice.

# In[ ]:


""" Load the dataset
Dataset (Numpy npz file)
|- features (Numpy.ndarray)
|- labels (Numpy.ndarray)

The data is also provided in csv format.
"""


def load_data(file_name='./dataset1.npz'):
    """ Load the Numpy npz format dataset 
    Args:
        file_name (string): name and path to the dataset (dataset1.npz, dataset2.npz, dataset3.npz)
    Returns:
        X (Numpy.ndarray): features
        y (Numpy.ndarray): 1D labels
    """
    import numpy as np
    data = np.load(file_name)
    X, y = data['features'], data['labels']
    return X, y


# Load dataset 1 by default
X, y = load_data()


# print(X.shape)
# print(y.shape)


# ### Skeleton codes:
# You should follow the structure of this code:

# In[ ]:


class Perceptron:
    # Implement your base learner here
    def __init__(self, learning_rate, max_iter, **kwargs):
        """ Initialize the parameters here 
        Args:
            learning_rate (float or a collection of floats): your learning rate
            max_iter (int): the maximum number of training iterations
            Other parameters of your choice

        Examples ToDos:
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        Try different initialization strategies (as required in Question 2.3)
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        return


    def fit(self, X, y, **kwargs):
        """ Implement the training strategy here
        Args:
            X (Numpy.ndarray, list, etc.): The training data
            y (Numpy.ndarray, list, etc.): The labels
            Other parameters of your choice

        Example ToDos:
        # for _ in range(self.max_iter):
        #     Update the parameters of Perceptron according to the learning rate (self.learning_rate) and data (X, y)
        """
        self.w = np.zeros((X.shape[1], 1))  # Initialize the weights randomly
        self.b = 1  # Initialize bias with 1
        self.any_false = True
        self.iter = 0
        # Count the number of samples not correctly classified
        while self.any_false and self.iter < self.max_iter:
            mis_classified_number = 0
            for i in range(X.shape[0]):
                X_i = X[i]
                y_i = y[i]
                classify_result = np.dot(self.w.T, X_i.T) + self.b
                if y_i * classify_result < 0:  # This means sample is misclassified
                    self.w += self.learning_rate * np.dot(X_i, y_i).reshape(2, 1)
                    self.b += self.learning_rate * y_i
                    mis_classified_number += 1
            if mis_classified_number == 0:
                self.any_false = False  # If there isn't any sample misclassified, then the loop will end
            else:
                self.any_false = True  # If there is still any misclassified sample, then the loop kees running
            self.iter += 1
        print(self.w)
        print(self.b)
        return self.w, self.b

    def predict(self, x, **kwargs) -> np.ndarray:
        """ Implement the prediction strategy here
        Args:
            x (Numpy.ndarray, list, Numpy.array, etc.): The input data
            Other parameters of your choice
        Return(s):
            The prediction value(s), namely, class label(s), others of your choice
        """
        labels = np.zeros(x.shape[0],1)
        for int, data_point in enumerate(x):
            labels[int] = np.dot(self.w.T, data_point.T) + self.b


        return labels


perceptron = Perceptron(0.1, 1000)


# In[ ]:


class BoostingAlgorithm:
    # Implement your boosting algorithm here
    def __init__(self, n_estimators, **kwargs):
        """ Initialize the parameters here 
        Args:
            n_estimators (int): number of base perceptron models
            Other parameters of your choice

        Think smartly on how to utilize multiple perceptron models
        """
        pass

    def fit(self, X, y, **kwargs):
        """ Implement the training strategy here
        Args:
            X (Numpy.ndarray, list, etc.): The training data
            y (Numpy.ndarray, list, etc.): The labels
            Other parameters of your choice
        """
        pass

    def predict(self, x, **kwargs):
        """ Implement the prediction strategy here
        Args:
            x (Numpy.ndarray, list, Numpy.array, etc.): The input data
            Other parameters of your choice
        Return(s):
            The prediction value, namely, class label(s)
        """
        return


# In[ ]:


def run(**kwargs):
    """ Single run of your classifier
    # Load the data
    X, y = load_data()
    # Find a way to split the data into training and test sets
    -> X_train, y_train, X_test, y_test

    # Initialize the classifier
    base = Perceptron("your parameters")

    # Train the classifier
    base.fit(X_train, y_train, "other parameters")

    # Test and score the base learner using the test data
    y_pred = base.predict(X_test, "other parameters")
    score = SCORING(y_pred, y_test)
    """
    pass

# Good luck with the assignment
