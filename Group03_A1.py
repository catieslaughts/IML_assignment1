import sys
#!{sys.executable} -m pip install numpy
import os
import numpy as np
import matplotlib.pyplot as plt

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
        self.w = np.random.rand(X.shape[1], 1)
        #self.w = np.zeros((X.shape[1], 1))  # Initialize the weights to zero
        self.b = 1  # Initialize bias with 1
        self.any_false = True
        self.iter = 0
        
        curr_lr = self.learning_rate
        
        # Count the number of samples not correctly classified
        while self.any_false and self.iter < self.max_iter:
            #print(curr_lr)
            mis_classified_number = 0
            for i in range(X.shape[0]):
                X_i = X[i]
                y_i = y[i]
                classify_result = np.dot(self.w.T, X_i.T) + self.b
                
                if y_i * classify_result < 0:  # This means sample is misclassified
                    self.w += curr_lr * np.dot(X_i, y_i).reshape(self.w.shape[0], self.w.shape[1])
                    self.b += curr_lr * y_i
                    mis_classified_number += 1
            if mis_classified_number == 0:
                self.any_false = False  # If there isn't any sample misclassified, then the loop will end
            else:
                self.any_false = True  # If there is still any misclassified sample, then the loop kees running
            self.iter += 1
            curr_lr = curr_lr/self.iter
        #print(self.w)
        #print(self.b)
        
        pass

    def predict(self, x, **kwargs) -> np.ndarray:
        """ Implement the prediction strategy here
        Args:
            x (Numpy.ndarray, list, Numpy.array, etc.): The input data
            Other parameters of your choice
        Return(s):
            The prediction value(s), namely, class label(s), others of your choice
        """ 
        labels = np.zeros(x.shape[0]).astype(int)
        for i, data_point in enumerate(x):
            temp = np.dot(self.w.T, data_point.T) + self.b
            if temp < 0:
                labels[i] = -1
            else:
                labels[i] = 1


        return labels

class BoostingAlgorithm:
    # Implement your boosting algorithm here
    def __init__(self, n_estimators, **kwargs):
        """ Initialize the parameters here 
        Args:
            n_estimators (int): number of base perceptron models
            Other parameters of your choice
        
        Think smartly on how to utilize multiple perceptron models
        """
        self.n_estimators = n_estimators
        self.alpha = np.empty(n_estimators)
        self.learners = np.empty(n_estimators, dtype=object)
        
        pass

    def fit(self, X, y, **kwargs):
        """ Implement the training strategy here
        Args:X (Numpy.ndarray, list, etc.): The training data
            y (Numpy.ndarray, list, etc.): The labels
            Other parameters of your choice
        """ 
        verbose = True
        learning_rate = 1 #starts at 1
        max_iter = 200 #maximum iterations
        
        if verbose:
        	print('Maximum number of iterations: '+str(max_iter))
        
        training_data_len = y.size / 2 #can be changed
        
        curr_w = np.ones(y.size)
        curr_w = curr_w/y.size #sum(curr_w) should equal 1
        
        curr_X = X.copy()
        curr_y = y.copy()
        
        
        sort_idx = np.arange(y.size) #creates array of indexes for sorting
        
        for i in range(self.n_estimators):
            self.learners[i] = Perceptron(learning_rate, max_iter)#create current learner
            
            idx_array = np.arange(curr_y.size)
            subset_idx = np.random.choice(idx_array, size=int(training_data_len), replace=False, p=curr_w)
            X_sub, y_sub = curr_X[subset_idx], curr_y[subset_idx]#randomly select subset to train, weighted
            
            self.learners[i].fit(X_sub, y_sub)#train learner
            output = self.learners[i].predict(curr_X) #predict
            
            wrong = np.asarray([int(j) for j in (curr_y != output)])
            
            if sum(wrong) == 0:
                err = np.dot(curr_w, wrong)/.000001 #avoids divide by zero errors
                self.alpha[i] = .5 * np.log((1 - err)/.000001)
            else:
                err = np.dot(curr_w, wrong)/sum(wrong)
                self.alpha[i] = .5 * np.log((1 - err)/err)
            
            update_sign = np.asarray([j if j == 1 else -1 for j in wrong])
            
            new_weights = curr_w * np.exp(update_sign * self.alpha[i])#update weights
            new_weights = new_weights / np.sum(new_weights) #normalize to 1
            
            curr_w = new_weights
        pass   
        #return self.learners

    def predict(self, x, **kwargs):
        """ Implement the prediction strategy here
        Args:
            x (Numpy.ndarray, list, Numpy.array, etc.): The input data
            Other parameters of your choice
        Return(s):
            The prediction value, namely, class label(s)
        """ 
        #print(self.learners)
        pred_sum = 0
        #print("alphas: "+str(self.alpha.size))
        for i, learner in enumerate(self.learners):
            test = learner.predict(x)
            pred_sum += test*self.alpha[i]
        
        pred_avg = pred_sum/self.n_estimators #the weighted average prediction
        
        prediction = np.asarray([-1 if i < 0 else 1 for i in pred_avg])
        
        return prediction
        
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
    verbose = True
    
    n_estimators = 50
    dataset = './dataset2.npz'
    
    X, y = load_data(dataset)
    if verbose:
    	print('Loading data from '+dataset)
    	print('')
    
    idx_array = np.arange(y.size)
    
    training_size = int(.6*y.size) #number of points to be selected for the training set
    train_idx = np.random.choice(idx_array, size=training_size, replace=False)
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = np.delete(np.copy(X), train_idx, axis = 0), np.delete(np.copy(y), train_idx, axis = 0)
    
    if verbose:
    	print('Training data set size: '+str(y_train.size))
    	print('Test data set size: '+str(y_test.size))
    	print('')
    	print('Number of base learners in ensemble: '+str(n_estimators))
    
    strong_learner = BoostingAlgorithm(n_estimators)
    
    strong_learner.fit(X_train, y_train)
    output = strong_learner.predict(X_test)
    
    loss = np.sum(np.not_equal(output,y_test))/output.size
    
    print('')
    print('Ensemble Loss = '+str(loss))
    
    #return X_test, y_test, output, all_learners, alphas
    pass
    
run()
