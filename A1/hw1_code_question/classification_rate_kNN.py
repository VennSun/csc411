from run_knn import*
from utils import *
import matplotlib.pyplot as plt


def classification_rate(targets, k, train_data, train_labels, data):

    count = 0

    valid_label = run_knn(k, train_data, train_labels, data)

    for i in range(len(valid_label)):
        if valid_label[i] == targets[i]:
            count += 1
    #print(count)
    #print(len(targets))
    return float(count) / (len(targets))

if __name__ == '__main__':

    train_data, train_labels = load_train()

    valid_data, valid_targets = load_valid()

    #nomally we won't do this, only for this question as indicated in the assignment.
    test_data, test_targets = load_test()

    k_values = [1, 3, 5, 7, 9]

    classification_rate_for_valid = []
    classification_rate_for_test = []

    for k in k_values:

        valid_classification_rate = classification_rate(valid_targets, k, train_data, train_labels, valid_data)

        test_classification_rate = classification_rate(test_targets, k, train_data, train_labels, test_data)

        classification_rate_for_valid.append(valid_classification_rate)

        classification_rate_for_test.append(test_classification_rate)

    #plot graph
    plt.plot(k_values, classification_rate_for_valid, label='Validation Set')
    plt.plot(k_values, classification_rate_for_test, label='Test Set')
    plt.xlabel('k values')
    plt.ylabel('classification rate')
    plt.legend()
    plt.title('Classification Rates for kNN')
    plt.show()

