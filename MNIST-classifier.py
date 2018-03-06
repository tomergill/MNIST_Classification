from mnist import MNIST
import xgboost as xgb
from sys import argv
from os import getcwd
from time import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def main(learning_type='simple', location=None, max_depth=-1, folds=3, num_boosts=10):
    """
    Main function. Loads the data, initializes the learning model, fits it to the training data,
        predicts on the test data and computes the accuracy. Prints the results and how much time it took.

    :param learning_type: A string describing the learning algorithm being used. The options are:
        * simple - uses xgboost's XGBClassifier
        * Decision Tree - uses sklearn DecisionTreeClassifier using max_depth
        * Cross Validation - uses xgboost's cv function using max_depth, folds as nfolds
               and num_boosts as num_boost_round

    :param location: The location of the train and test data. If none is porvided will use the current folder.

    :param max_depth: The maximal depth of the decision tree(s), if any are used.

    :param folds: The k of the k-fold-cross-validation.

    :param num_boosts: How many different decision trees will be combined (boosting).

    :return: None
    """

    # Load data in a numpy array
    mndata = MNIST(location if location is not None else getcwd())
    images, labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    images, labels = np.array(images), np.array(labels)
    test_images, test_labels = np.array(test_images), np.array(test_labels)

    # Initializes the model
    if learning_type == 'Decision Tree':
        if max_depth < 1:
            print "ERROR: Max Depth must be 1 or greater"
            return
        model = DecisionTreeClassifier(max_depth=max_depth)

    elif learning_type == 'Cross Validation':
        # adds the train and test sets to one
        dmat = xgb.DMatrix(np.concatenate((images, test_images)), label=np.concatenate((labels, test_labels)))
        parmas = {"max_depth": max_depth, "num_class": 10, 'seed': 123, 'eta': 0.1}

    else:
        model = xgb.XGBClassifier(n_estimators=30, objective="reg:logistic", learning_rate=1.5)

    # Training the model and predicting
    if learning_type == 'Cross Validation':
        print "Training & Predicting..."
        start = time()
        results = xgb.cv(parmas, dmat, num_boosts, folds, metrics="error", seed=123)
        print "Finished predicting in {} seconds, with {}% accuracy.".format(
            time() - start, 1 - results[-1, 1] * 100)

    else:  # XGBClassifier and DecisionTreeLearner
        print "Training..."
        start = time()
        model.fit(images, labels)
        print "Finished training in {} seconds.\nStarting predicting...".format(time() - start)

        start = time()
        preds = model.predict(test_images)
        print "Finished predicting in {} seconds, with {}% accuracy.".format(
            time() - start, float(np.sum(preds == test_labels)) / test_labels.shape[0] * 100)


"""
See README for command line arguments.
"""
if __name__ == '__main__':

    learning_type = 'simple'  # uses XGBClassifier
    max_depth = -1
    location = argv[1] if len(argv) > 1 else None
    folds = -1
    boosts = -1

    if len(argv) > 1 and argv[1] == 'dt':  # uses DecisionTreeClassifier
        learning_type = 'Decision Tree'
        max_depth = int(argv[2])
        max_depth = max_depth if max_depth > 0 else None
        location = argv[3] if len(argv) > 3 else None

    elif len(argv) > 1 and argv[1] in ['boosting', 'cv']:  # uses xgboost.cv()
        learning_type = 'Cross Validation'
        max_depth = int(argv[2])
        max_depth = max_depth if max_depth > 0 else None
        folds = int(argv[3])
        boosts = int(argv[4])
        location = argv[5] if len(argv) > 5 else None

    main(learning_type=learning_type, location=location, max_depth=max_depth, folds=folds, num_boosts=boosts)
