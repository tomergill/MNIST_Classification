from mnist import MNIST
import xgboost as xgb
from sys import argv
from os import getcwd
from time import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def main(learning_type='simple', location=None, max_depth=-1, folds=-1, num_boosts=-1):
    mndata = MNIST(location if location is not None else getcwd())
    images, labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    images, labels = np.array(images), np.array(labels)
    test_images, test_labels = np.array(test_images), np.array(test_labels)

    if learning_type == 'Decision Tree':
        model = DecisionTreeClassifier(max_depth=max_depth)
    elif learning_type == 'Cross Validation':
        # dtrain, dtest = xgb.DMatrix(images, label=labels), xgb.DMatrix(test_images, label=test_labels)
        dmat = xgb.DMatrix(np.concatenate((images, test_images)), label=np.concatenate((labels, test_labels)))
        parmas = {"max_depth": max_depth, "objective": "multi:softmax", "num_class": 10}
    else:
        model = xgb.XGBClassifier(n_estimators=10, objective="reg:logistic")

    if learning_type == 'Cross Validation':
        print "Training & Predicting..."
        start = time()
        results = xgb.cv(parmas, dmat, num_boosts, folds, metrics="error", seed=123)
        print "Finished predicting in {} seconds, with {}% accuracy.".format(
            time() - start, 1-results[-1, 1] * 100)

    else:
        print "Training..."
        start = time()
        model.fit(images, labels)
        print "Finished training in {} seconds.\nStarting predicting...".format(time() - start)

        start = time()
        preds = model.predict(test_images)
        print "Finished predicting in {} seconds, with {}% accuracy.".format(
            time() - start, float(np.sum(preds == test_labels)) / test_labels.shape[0] * 100)


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

    elif len(argv) > 1 and argv[1] in ['boosting', 'cv']:
        learning_type = 'Cross Validation'
        max_depth = int(argv[2])
        max_depth = max_depth if max_depth > 0 else None
        folds = int(argv[3])
        boosts = int(argv[4])
        location = argv[5] if len(argv) > 5 else None

    main(learning_type=learning_type, location=location, max_depth=max_depth, folds=folds, num_boosts=boosts)
