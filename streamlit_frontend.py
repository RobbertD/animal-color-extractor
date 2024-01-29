from pathlib import Path
import tempfile
import time
from zipfile import ZipFile
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from tempfile import NamedTemporaryFile

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

def scale(image, scale_percent):
    # Calculate the new dimensions
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)

    # Resize the image
    new_dimensions = (width, height)
    resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

    return resized_image
    
def create_zip_archive(images_folder):
    # create unique filename
    zip_filename = time.strftime("%Y%m%d%H%M%S") + ".zip"
    images_folder_path = Path(images_folder)
    zip_filename_path = Path(zip_filename)

    with ZipFile(zip_filename_path, 'w') as zipf:
        for file_path in images_folder_path.iterdir():
            zipf.write(file_path, arcname=file_path.name)
    
    return zip_filename_path

def main():
    st.title("Animal color extractor App")
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # put both images side by side
        min_area = st.slider("Min Area", 100, 10000, 5000)
        # cols = st.columns(2)
        # with cols[0]:
        #     st.image(image, caption="Uploaded Image", use_column_width=False)
        
        # Detect and cut objects with adjustable parameters
        thr, objects_data, masks = detect_objects_threshold(image, 0, min_area)

        # with cols[1]:
            # st.image(thr, caption="thr Image", use_column_width=False)
        
        # Display results
        st.subheader("Detected Objects:")
        overlay = overlay_objects_threshold(image, objects_data)
        st.image(overlay, caption="Uploaded Image", use_column_width=False)
        # st.write(pd.DataFrame(objects_data))

        #Show cutouts of detected obj
        cutouts = cutout(objects_data, image)
        masked_cutouts = []
        for i, _cutout in enumerate(cutouts):
            with st.container(border=True):
                st.subheader(f"Cutout {i+1}")
                name = st.text_input("Name", value=f"cutout_{i+1}", key=f"cutout_name_{i+1}")
                cols = st.columns(4)
                
                with cols[3]:
                    include = st.checkbox("Include", value=True, key=f"include_{i}")
                    cutout_threshold = st.slider("Min Threshold", 0, 255, 230, key=f"min_threshold_{i}")
                    fill_hole_input = st.checkbox("Fill holes", value=True, key=f"fill_hole_input_{i}")
                    erode_input = st.checkbox("Erode", value=True, key=f"erode_input_{i}")
                    erode_level = st.slider("Erode Level", 1, 10, 5, key=f"erode_level_{i}")

                _, obj_data, masks = detect_objects_threshold(_cutout, cutout_threshold, min_area, max_num_objects=1, thresholding_func=cv2.THRESH_BINARY_INV)
                mask = masks[0].astype(np.uint8)*255
                
                if fill_hole_input:
                    mask = fill_holes(mask)
                # erode
                if erode_input:
                    ele = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_level, erode_level))
                    mask = cv2.erode(mask, ele)

                # masked cutout on green basckground
                masked_cutout = cv2.bitwise_and(_cutout, _cutout, mask=mask)
                masked_cutout[mask == 0] = (0, 255, 0)

                
                with cols[0]:
                    st.image(_cutout, caption=f"Cutout", use_column_width=False)
                with cols[1]:
                    st.image(mask, caption=f"Cutout mask", use_column_width=False)
                # apply mask
                with cols[2]:
                    st.image(masked_cutout, caption=f"Masked Cutout", use_column_width=False)
                
                if include:
                    masked_cutouts.append((name,masked_cutout))
        
        st.subheader("Download Results:")
        zip_path = None
        if st.button("Zip images"):
            # Create a temporary folder
            temp_folder = tempfile.TemporaryDirectory()
            temp_folder_path = Path(temp_folder.name)

            for i, (name, _cutout) in enumerate(masked_cutouts):
                _cutout = cv2.cvtColor(_cutout, cv2.COLOR_BGR2RGB)
                cv2.imwrite(f"{temp_folder_path}/{name}.png", _cutout)
            
            zip_path = create_zip_archive(temp_folder_path)
            
            # Delete the temporary folder when done
            temp_folder.cleanup()

        if zip_path:
            with open(zip_path, "rb") as fp:
                btn = st.download_button(
                    label="Download ZIP",
                    data=fp,
                    file_name=zip_path.name,
                    mime="application/zip"
                )
            #remove zip 
            zip_path.unlink()
        
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
