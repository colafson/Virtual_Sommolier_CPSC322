import mysklearn.myutils as myutils
import numpy as np
import operator
import random

class MySimpleLinearRegressor:
    """Represents a simple linear regressor.

    Attributes:
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b

    Notes:
        Loosely based on sklearn's LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    
    def __init__(self, slope=None, intercept=None):
        """Initializer for MySimpleLinearRegressor.

        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
        """
        self.slope = slope 
        self.intercept = intercept

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train) 
                The shape of y_train is n_train_samples
        """
        meanX = sum(X_train)/len(X_train)
        meanY = sum(y_train)/len(y_train)
        m = 0
        top = 0 
        bottomX = 0
        bottomY = 0 
        for i in range(len(X_train)):
            top +=  ((X_train[i]-meanX)*(y_train[i]-meanY))
            bottomX += (X_train[i]-meanX)**2
            bottomY += (y_train[i]-meanY)**2
        m = top/bottomX
        b = meanY-m*meanX
        self.slope = m
        self.intercept = b
        return m,b

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]

        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        predictions = []
        for i in X_test:
            predictions.append((self.slope*i)+self.intercept)
        return predictions


class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train 
        self.y_train = y_train 

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances 
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        for i, instance in enumerate(self.X_train[0]):
            instance.append(self.y_train[i])
            instance.append(int(i))
            distance = myutils.compute_euclidean_distance(instance[:len(X_test[0])], X_test[0])
            instance.append(distance)
        sort = sorted(self.X_train[0], key=operator.itemgetter(-1) )
        closest = sort[:self.n_neighbors]
        dist = []
        neighbors = []
        for x in closest:
            dist.append(x[-1])
            neighbors.append(x[-2])
        return dist, neighbors 

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        results = self.kneighbors(X_test)
        values = []
        for index in results[1]:
            values.append(self.X_train[0][index][-3])
        unique = []
        scores = []
        for entry in values:
            if entry in unique:
                ind = values.index(entry)
                scores[ind] += 1
            else:
                unique.append(entry)
                scores.append(1)
        maximum = max(scores)
        index = scores.index(maximum)
        return [unique[index]] 


class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.priors = None 
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        self.X_train = X_train
        self.y_train = y_train

        #make prior a 2d array with first value being class and second is ratio
        priors = []
        for x in y_train:
            inPriors = False
            for elements in priors:
                if elements[0] == int(x):
                    elements[1] += 1
                    inPriors = True
            if not inPriors:
                priors.append([x,1])
        #posteriors 
        posteriors = []
        for col in range(len(X_train[0])):
            attributes = []
            for row in range(len(X_train)):
                isIn = False
                for element in attributes:
                    if X_train[row][col] == element[0]:
                        class_val = int(y_train[row])
                        element[class_val+1] += 1
                        isIn = True
                if not isIn:
                    val = [X_train[row][col]]
                    for _ in range(len(priors)):
                        val.append(0)
                    class_val = int(y_train[row])
                    val[class_val+1] += 1
                    attributes.append(val)
            for classes in priors:
                for elements in attributes:
                    elements[classes[0]+1] = elements[classes[0]+1] / classes[1]     
            posteriors.append(attributes)
        self.posteriors = posteriors

        for elements in priors:
            elements[1] = elements[1]/len(y_train)
        self.priors = priors


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        results = []
        for test in X_test:
            values = []
            for attribute in range(len(test)):
                for vals in range(len(self.posteriors[attribute])):
                    if test[attribute] == self.posteriors[attribute][vals][0]:
                        if len(values) == 0:
                            values = self.posteriors[attribute][vals][1:]
                        else:
                            for elms in range(len(values)):
                                values[elms] = values[elms] * self.posteriors[attribute][vals][elms+1]
            for x in range(len(self.priors)):
                values[self.priors[x][0]] = values[self.priors[x][0]]*self.priors[x][1]
            better = max(values)
            win_val = values.index(better)
            results.append(win_val)
        return results


class MyZeroRClassifier:

    def __init__(self):
        self.X_train = None 
        self.y_train = None
        self.priors = None 
        self.posteriors = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        priors = []
        for x in y_train:
            inPriors = False
            for elements in priors:
                if elements[0] == x:
                    elements[1] += 1
                    inPriors = True
            if not inPriors:
                priors.append([x,1])
        for elements in priors:
            elements[1] = elements[1]/len(y_train)
        self.priors = priors

    def predict(self):
        max_val = self.priors[0]
        for x in self.priors:
            if x[1] > max_val[1]:
                max_val = x
        return max_val[0]

class MyRandomClassifier:
    def __init__(self):
        self.X_train = None 
        self.y_train = None
        self.priors = None 
        self.posteriors = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        priors = []
        for x in y_train:
            inPriors = False
            for elements in priors:
                if elements[0] == x:
                    elements[1] += 1
                    inPriors = True
            if not inPriors:
                priors.append([x,1])
        for elements in priors:
            elements[1] = elements[1]/len(y_train)
        self.priors = priors

    def predict(self):
        index = random.randrange(len(self.priors))
        return self.priors[index][0]

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train, F):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train

        header = myutils.make_header(self.X_train)
        attribute_domains = myutils.make_att_domain(self.X_train, header)

        # my advice: stitch together X_train and y_train
        train = [X_train[i] + [y_train[i]] for i in range(0,len(y_train))]
        available_attributes = header.copy() 
        # initial tdidt() call
        #normal tree function
        #self.tree = myutils.tdidt(train, available_attributes,header,attribute_domains)
        #random forest function
        self.tree = myutils.random_forest_tdidt(train, available_attributes,header,attribute_domains, F)
        
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        result = []
        for test in X_test:
            val = myutils.classify_tdidt(self.tree, test)
            result.append(val)
        return result

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        tree = self.tree
        if attribute_names == None:
            attribute_names = myutils.make_header(self.X_train)
        myutils.print_rules(tree, attribute_names, class_name, "")


class MyRandomForestGenerator:
        """Represents a decision tree classifier.
        Attributes:
            X_train(list of list of obj): The list of training instances (samples). 
                    The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train). 
                The shape of y_train is n_samples
            tree(nested list): The extracted tree model.

        Notes:
            Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
            Terminology: instance = sample = row and attribute = feature = column
        """
        def __init__(self):
            """Initializer for MyDecisionTreeClassifier.

            """
            self.X_train = None 
            self.y_train = None
            self.best_trees = []

        def fit(self, X_train, y_train, N, M, F):
            """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

            Args:
                X_train(list of list of obj): The list of training instances (samples). 
                    The shape of X_train is (n_train_samples, n_features)
                y_train(list of obj): The target y values (parallel to X_train)
                    The shape of y_train is n_train_samples

            Notes:
                Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                    from the training data.
                Build a decision tree using the nested list representation described in class.
                Store the tree in the tree attribute.
                Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
            """
            self.X_train = X_train
            self.y_train = y_train
 
            trees = []
            for _ in range(N):
                tree_entry = []
                remainder_set, validation_set = myutils.bootstrap_sample(X_train, y_train)
                #construct trees 
                header = myutils.make_header(self.X_train)
                attribute_domains = myutils.make_att_domain(self.X_train, header)
                # my advice: stitch together X_train and y_train
                train = [X_train[i] + [y_train[i]] for i in range(0,len(y_train))]
                available_attributes = header.copy() 
                # initial tdidt() call
                new_tree = myutils.random_forest_tdidt(train, available_attributes,header,attribute_domains, F)
                #add tree to trees
                tree_entry.append(new_tree)
                tree_answers = []
                for x in validation_set[0]:
                    tree_answers.append(myutils.classify_tdidt(new_tree, x))

                accuracy = 0
                for x in range(len(tree_answers)):
                    if tree_answers[x] == validation_set[1][x]:
                        accuracy+=1
                accuracy = accuracy/len(tree_answers)
                #add accuracy to the tree array
                tree_entry.append(accuracy)
                trees.append(tree_entry)
            #step 3
            for tree in trees:
                #compute accuracy based of of the 
                if len(self.best_trees) < M:
                    #add it to best_trees array
                    self.best_trees.append(tree)
                else:
                    for element in self.best_trees:
                        if element[1] < tree[1]:
                            index = self.best_trees.index(element)
                            self.best_trees[index] = tree 
                            break
            
        def predict(self, X_test):
            """Makes predictions for test instances in X_test.

            Args:
                X_test(list of list of obj): The list of testing samples
                    The shape of X_test is (n_test_samples, n_features)

            Returns:
                y_predicted(list of obj): The predicted target y values (parallel to X_test)
            """
            tree_answers = []
            for x in self.best_trees:
                new_tree = x[0]
                answers = []
                for x in X_test:
                    answers.append(myutils.classify_tdidt(new_tree, x))
                tree_answers.append(answers)
            
            answers = []
            for index in range(len(tree_answers[0])):
                values = []
                for row in tree_answers:
                    values.append([row[index]])
                result = myutils.majority_rule(values)
                answers.append(result)
            return answers