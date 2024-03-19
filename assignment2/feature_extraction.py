# This assignment covers keypoint matching and image stitching with SIFT and RANSAC.
# Alyssa Hammonds
 
import numpy as np
import cv2
import pickle
from skimage.feature import hog
from sklearn.cluster import KMeans

def create_bag_of_words(features, clusters):

    flattened_features = np.concatenate(features)
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(flattened_features)

    return kmeans.cluster_centers_

def generate_histograms(features, words):
    histograms = []
    for image_features in features:
        labels = KMeans.predict(image_features)
        histogram, _ = np.histogram(labels, bins=range(len(words) + 1))
        histogram = histogram.astype(float) / np.sum(histogram)
        histograms.append(histogram)

    return histograms


def extract_sift_features(data):
    sift = cv2.SIFT_create()
    features = []
    for image in data:
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Extract SIFT features
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        features.append(descriptors)
    return features

def extract_hog_features(data):
    features = []
    for image in data:
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        fd, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys',
                            visualize=True, feature_vector=True)
        features.append(fd)
    return features

def save_features(features, labels, filename):
    data = {
        "features": features,
        "labels": labels
    }
    with open(filename, "wb") as file:
        pickle.dump(data, file)

def to_img(data):
        # reshape the data to 32x32 images and uint8
        return data.reshape(-1, 32, 32, 3).astype(np.uint8)

if __name__ == "__main__":
    # Load the pre-split data
    data = np.load("cifar10.npz", allow_pickle=True)

    # Extract features from the training data
    x_train = data["X_train"]
    train_img = to_img(x_train)
    y_train = data["y_train"]

    # Extract features from the testing data
    x_test = data["X_test"]
    test_img = to_img(x_test)
    y_test = data["y_test"]

    # Extract SIFT features
    train_sift = extract_sift_features(train_img)
    test_sift = extract_sift_features(test_img)

    # Concatenate SIFT features
    all_sift = train_sift + test_sift

    # Save the SIFT features
    save_features(train_sift, y_train, "train_sift_features.pkl")
    save_features(test_sift, y_test, "test_sift_features.pkl")

    # Extract HOG features
    train_hog = extract_hog_features(train_img)
    test_hog = extract_hog_features(test_img)

    # Save the HOG features
    save_features(train_hog, y_train, "train_hog_features.pkl")
    save_features(test_hog, y_test, "test_hog_features.pkl")

    clusters = 100
    words = create_bag_of_words(all_sift, clusters)
    train_histograms = generate_histograms(train_sift, words)

    # Generate histograms for the testing data
    test_histograms = generate_histograms(test_sift, words)

    # Save the histograms
    save_features(train_histograms, y_train, "train_histograms.pkl")
    save_features(test_histograms, y_test, "test_histograms.pkl")


    