from pathlib import Path
import tempfile
import time
from zipfile import ZipFile
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from tempfile import NamedTemporaryFile
import utils

import cv2
import numpy as np



def main():
    st.title("Animal color extractor App")
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        min_area = st.slider("Min Area", 100, 10000, 5000)
        
        # Detect and cut objects with adjustable parameters
        thr, objects_data, masks = utils.detect_objects_threshold(image, 0, min_area)

        # Display results
        st.subheader("Detected Objects:")
        overlay = utils.overlay_objects_threshold(image, objects_data)
        st.image(overlay, caption="Uploaded Image", use_column_width=False)

        #Show cutouts of detected obj
        cutouts = utils.cutout(objects_data, image)
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

                _, obj_data, masks = utils.detect_objects_threshold(_cutout, cutout_threshold, min_area, max_num_objects=1, thresholding_func=cv2.THRESH_BINARY_INV)
                mask = masks[0].astype(np.uint8)*255
                
                if fill_hole_input:
                    mask = utils.fill_holes(mask)
                # erode
                if erode_input:
                    ele = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_level, erode_level))
                    mask = cv2.erode(mask, ele)

                # masked cutout on green basckground
                masked_cutout = cv2.bitwise_and(_cutout, _cutout, mask=mask)
                masked_cutout[mask == 0] = (0, 255, 0)

                
                with cols[0]:
                    st.image(_cutout, caption=f"Cutout", use_column_width=True)
                with cols[1]:
                    st.image(mask, caption=f"Cutout mask", use_column_width=True)
                # apply mask
                with cols[2]:
                    st.image(masked_cutout, caption=f"Masked Cutout", use_column_width=True)
                
                if include:
                    masked_cutouts.append((name,masked_cutout))

        
        zip_path = None
        # Create a temporary folder
        temp_folder_path = Path("temp_folder")
        temp_folder_path.mkdir(exist_ok=True)
        
        st.subheader("Calculate Kmeans")
        st.info("Note: This process can take a while. Parameters: https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html")
        if 'kmeans' not in st.session_state:
            st.session_state.kmeans = []
        n_clusters = st.number_input("Number of clusters", 1, 256, 256)
        eps = st.number_input("Epsilon", 0.0, 10.0, 1.0)
        max_iter = st.number_input("Max Iterations", 1, 100, 10)
        attempts = st.number_input("Attempts", 1, 100, 5)
        
        if st.button("Calculate Kmeans"):
            st.session_state.kmeans = []
            with st.spinner("Calculating Kmeans..."):
                for name, _cutout in masked_cutouts:
                    cluster_centers_with_pixels = utils.kmeans(_cutout, n_clusters, eps, max_iter=max_iter, attempts=attempts)
                    st.session_state.kmeans.append(cluster_centers_with_pixels)
            st.success("Done!")

        st.subheader("Download Results:")
        if st.button("Zip images and Kmeans csv"):

            for i, (name, _cutout) in enumerate(masked_cutouts):
                _cutout = cv2.cvtColor(_cutout, cv2.COLOR_BGR2RGB)
                cv2.imwrite(str(temp_folder_path / f"{name}.png"), _cutout)

                if len(st.session_state.kmeans)>0:
                    csv_file = temp_folder_path / f"{name}.csv"
                    with csv_file.open("w") as f:
                        # np array to csv
                        header = "Red, Green, Blue, Percentage\n"
                        np.savetxt(f, st.session_state.kmeans[i], delimiter=",", header=header)
            
            zip_path = utils.create_zip_archive(temp_folder_path)
            
            # Delete the temporary folder when done
            for a in temp_folder_path.iterdir(): a.unlink() 

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
