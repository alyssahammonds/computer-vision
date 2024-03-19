from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np


if __name__ == "__main__":
    # Load the data
    data = np.load("train_hog_features.pkl", allow_pickle=True)
    train_histogram = data['X_train'] 
    test_histogram = data['X_test']
    train_labels = data['y_train']
    test_labels = data['y_test']

    # create a linear SVM
    clf = svm.SVC(kernel='linear')

    # train the SVM
    clf.fit(train_histogram, train_labels)

    # predict the labels
    labels = clf.predict(test_histogram)

    # calculate the accuracy
    accuracy = accuracy_score(test_labels, labels)

    print(f"Accuracy: {accuracy:.2f}")


