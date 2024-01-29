from pathlib import Path
import time
from zipfile import ZipFile
import cv2
import numpy as np

def detect_objects_threshold(image, threshold_value, min_size,  max_num_objects=100, thresholding_func=cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresholded = cv2.threshold(gray, threshold_value, 255, thresholding_func)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded, connectivity=8)

    # Extract data for each connected component (excluding background)
    objects_data = []
    masks = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        # You can add more conditions to filter objects based on size, aspect ratio, etc.
        if area > min_size:  # Adjust the minimum area threshold as needed
            objects_data.append({
                'Object Number': i,
                'X': x + w // 2,
                'Y': y + h // 2,
                'Width': w,
                'Height': h,
                'Area': area,
            })
            masks.append(labels == i)

    # Sort the objects by area
    objects_data = sorted(objects_data, key=lambda obj: obj['Area'])

    # Limit the number of objects
    objects_data = objects_data[:max_num_objects]

    return thresholded, objects_data, masks

def cutout(objects_data, image):
    cutouts = []
    for objd in objects_data:
        x, y = int(objd['X']), int(objd['Y'])
        w, h = int(objd['Width'] // 2), int(objd['Height'] // 2)
        cutout = image[y - h:y + h, x - w:x + w]
        cutouts.append(cutout)
    return cutouts


def overlay_objects_threshold(original_image, objects_data):
    # Make a copy of the original image to avoid modifying the original
    overlay_image = original_image.copy()

    # Loop through the objects and draw rectangles on the overlay image
    for obj in objects_data:
        x, y = int(obj['X']), int(obj['Y'])
        w, h = int(obj['Width'] // 2), int(obj['Height'] // 2)

        # Draw a rectangle around the object
        cv2.rectangle(overlay_image, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

    return overlay_image

def fill_holes(binary_mask):
    # Create a copy of the mask to avoid modifying the original
    filled_mask = binary_mask.copy()

    # Find contours in the binary mask
    contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fill each contour in the mask
    for contour in contours:
        cv2.drawContours(filled_mask, [contour], 0, 255, thickness=cv2.FILLED)

    return filled_mask


def create_zip_archive(images_folder):
    # create unique filename
    zip_filename = time.strftime("%Y%m%d%H%M%S") + ".zip"
    images_folder_path = Path(images_folder)
    zip_filename_path = Path(zip_filename)

    with ZipFile(zip_filename_path, 'w') as zipf:
        for file_path in images_folder_path.iterdir():
            print(file_path)
            zipf.write(file_path, arcname=file_path.name)
    
    return zip_filename_path

def kmeans(image, clusters, eps, max_iter, attempts):
    Z = image.reshape((-1,3))

    #  filter out the green pixels
    Z = Z[~np.all(Z == (0, 255, 0), axis=1)]

    # convert to np.float32
    Z = np.float32(Z)

    # Apply k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)
    flags = cv2.KMEANS_RANDOM_CENTERS

    compactness, labels, centers = cv2.kmeans(
        Z, K=clusters, bestLabels=None, criteria=criteria, attempts=attempts, flags=flags
    )

    # Count the number of pixels in each cluster
    num_pixels = np.bincount(labels.flatten())

    # to percentages
    percentage_pixels = num_pixels / num_pixels.sum()

    # Concatenate the cluster centers with the number of pixels
    cluster_centers_with_percentages = np.column_stack((centers, percentage_pixels))

    return cluster_centers_with_percentages
