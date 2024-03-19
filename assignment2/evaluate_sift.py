import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC


if __name__ == "__main__":
    # Load the data from feature_extraction.py
    data = np.load("train_sift_features.pkl", allow_pickle=True)
    train_histogram = data['X_train']
    test_histogram = data['X_test']
    train_labels = data['y_train']
    test_labels = data['y_test']

    # create a linear svm
    clf = LinearSVC()

    # train the svm
    clf.fit(train_histogram, train_labels)

    # predict the labels
    labels = clf.predict(test_histogram)

    # calculate the accuracy
    accuracy = np.mean(labels == test_labels)

    print(f"Accuracy: {accuracy:.2f}")



