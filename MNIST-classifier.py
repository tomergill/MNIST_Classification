from mnist import MNIST
import xgboost as xgb
from sys import argv
from os import getcwd
from time import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def main(learning_type='simple', location=None, max_depth=-1):
    mndata = MNIST(location if location is not None else getcwd())
    images, labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    images, labels = np.array(images), np.array(labels)
    test_images, test_labels = np.array(test_images), np.array(test_labels)

    if learning_type == 'Decision Tree':
        model = DecisionTreeClassifier(max_depth=max_depth)
    else:
        model = xgb.XGBClassifier(n_estimators=10, objective="logistic:multiple")

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
    location = argv[2] if len(argv) > 2 else None

    if argv[1] == 'dt':  # uses DecisionTreeClassifier
        learning_type = 'Decision Tree'
        max_depth = int(argv[2])
        max_depth = max_depth if max_depth > 0 else None
        location = argv[3] if len(argv) > 3 else None

    main(learning_type=learning_type, location=location, max_depth=max_depth)
