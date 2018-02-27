from mnist import MNIST
import xgboost as xgb
from sys import argv
from os import getcwd
from time import time
import numpy as np


def main(location=None):
    mndata = MNIST(location if location is not None else getcwd())
    images, labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    images, labels = np.array(images), np.array(labels)
    test_images, test_labels = np.array(test_images), np.array(test_labels)

    xg_cl = xgb.XGBClassifier(n_estimators=10, objective="logistic:multiple")

    print "Training..."
    start = time()
    xg_cl.fit(images, labels)
    print "Finished training in {} seconds.\nStarting predicting...".format(time() - start)

    start = time()
    preds = xg_cl.predict(test_images)
    print "Finished predicting in {} seconds, with {}% accuracy.".format(
        time() - start, float(np.sum(preds==test_labels))/test_labels.shape[0] * 100)


if __name__ == '__main__':
    if len(argv) > 1:
        main(argv[1])
    else:
        main()
