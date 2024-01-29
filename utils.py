import cv2
import numpy as np


def kmeans(image, clusters):
    Z = image.reshape((-1,3))

    #  filter out the green pixels
    Z = Z[~np.all(Z == (0, 255, 0), axis=1)]

    # convert to np.float32
    Z = np.float32(Z)

    # Apply k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
    attemps = 1
    flags = cv2.KMEANS_RANDOM_CENTERS

    compactness, labels, centers = cv2.kmeans(
        Z, K=clusters, bestLabels=None, criteria=criteria, attempts=attemps, flags=flags
    )

    # Count the number of pixels in each cluster
    num_pixels = np.bincount(labels.flatten())

    # to percentages
    percentage_pixels = num_pixels / num_pixels.sum()

    # Concatenate the cluster centers with the number of pixels
    cluster_centers_with_percentages = np.column_stack((centers, percentage_pixels))

    return cluster_centers_with_percentages
