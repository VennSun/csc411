from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import cv2 as cv
import os.path


def read_images():
    path_train = '/home/venn/Desktop/csc411/A3/411a3/train/'
    data_train = np.array([cv.imread(name) for name in os.listdir(path_train)], dtype=np.float64)

    path_val = '/home/venn/Desktop/csc411/A3/411a3/val/'
    data_val = np.array([cv.imread(name) for name in os.listdir(path_val)], dtype=np.float64)

    path_test = '/home/venn/Desktop/csc411/A3/411a3/test/'
    data_test = np.array([cv.imread(name) for name in os.listdir(path_test)], dtype=np.float64)

    return data_train, data_val, data_test


def read_csv():
    train_path = '/home/venn/Desktop/csc411/A3/411a3/train.csv'
    train_csv = np.genfromtxt(train_path, delimiter=",")
    train_target = train_csv[1:, 1]

    path = '/home/venn/Desktop/csc411/A3/411a3/sample_submission.csv'
    csv = np.genfromtxt(path, delimiter=",")
    val_target = csv[1:501, 1]
    test_target = csv[501:971, 1]

    return train_target, val_target, test_target




def saveToNpz():
    data_train, data_val, data_test = read_images()
    train_target, val_target, test_target = read_csv()
    np.savez('../train.npz',inputs_train = data_train, target_train = train_target,
             inputs_valid  = data_val,  target_valid  = val_target,
             inputs_test = data_test, target_test = test_target)


if __name__ == '__main__':
    saveToNpz()

