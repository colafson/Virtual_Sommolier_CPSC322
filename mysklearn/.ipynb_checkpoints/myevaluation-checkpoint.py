import mysklearn.myutils as myutils
import random
import numpy as np
import math

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets (sublists) based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before splitting

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    
    Note:
        Loosely based on sklearn's train_test_split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if random_state is not None:
        random.seed(random_state)
    if shuffle: 
        newX = X
        newY = y
        for i in range(len(X)):
            index = random.randrange(0,len(X))
            newX[i], newX[index] = newX[index], newX[i]
            newY[i], newY[index] = newY[index], newY[i]
        X = newX
        y = newY
    randX = X
    randY = y
    if isinstance(test_size, float):
        split_index = int(len(X)*(1-test_size))
    else:
        split_index = len(X)-test_size
    return randX[:split_index], randX[split_index:], randY[:split_index], randY[split_index:]

def kfold_cross_validation(X, n_splits=5):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.

    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes: 
        The first n_samples % n_splits folds have size n_samples // n_splits + 1, 
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    split_range = math.ceil(len(X)/n_splits)
    test_data = []
    train_data = []
    start_test = 0
    for index in range(n_splits):
        test = []
        train = []
        for index in range(len(X)):
            if index >= start_test and index < start_test + split_range:
                test.append(index)
            else:
                train.append(index)
        test_data.append(test)
        train_data.append(train)
        start_test = start_test + split_range
    return train_data, test_data

def stratified_kfold_cross_validation(X, y, n_splits=5):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples). 
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X). 
            The shape of y is n_samples
        n_splits(int): Number of folds.
 
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes: 
        Loosely based on sklearn's StratifiedKFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    classification = []
    indexes = []
    for x in range(len(X)):
        if y[x] in classification:
            index = classification.index(y[x])
            indexes[index].append(x)
        else:
            classification.append(y[x])
            indexes += [[x]]
    partioned = []
    for _ in range(n_splits):
        partioned.append([])

    count = 0
    for classes in range(len(classification)):
        for index in indexes[classes]:  
            partioned[count].append(index)
            count+=1
            count = count%len(partioned)
    X_train = []
    X_test = []
    
    for x in range(n_splits):
        test = []
        train = []
        for index in range(len(partioned)):
            if index == x:
                test+=partioned[index]
            else:
                train += partioned[index]
        X_train.append(train)
        X_test.append(test)
    return X_train, X_test
    

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry 
            indicates the number of samples with true label being i-th class 
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = []
    row = []
    for x in range(len(labels)):
        row = []
        for x in range(len(labels)):
            row.append(0)
        matrix.append(row)
    
    for x in range(len(y_true)):
        i1 = labels.index(y_true[x])
        i2 = labels.index(y_pred[x])
        matrix[i1][i2] +=1
    return matrix 
