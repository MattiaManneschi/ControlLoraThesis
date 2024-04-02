import time
import zipfile
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from streamlit_option_menu import option_menu

import backend

buf = BytesIO()

st.set_page_config(layout="wide")

appMode = st.sidebar.selectbox("App mode: ", ("inference", "train"))

if appMode == "inference":
    st.markdown("<h1 style='text-align: center; color: white;'>Welcome to SnapArt - Inference Session</h1>",
                unsafe_allow_html=True)
    modelType = option_menu("Choose the model", ["Bedside Lamp Model", "Desk Lamp Model", "Table Lamp Model"],
                            icons=["moon", "pen", "lamp"],
                            menu_icon="window", default_index=0, orientation="horizontal")
    adapterType = option_menu("Choose the adapter", ["CannyEdge", "Scribble"],
                              icons=["border", "pencil"],
                              menu_icon="controller", default_index=0, orientation="horizontal")
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
    )
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 10, 1)
    if drawing_mode == 'point':
        point_display_radius = st.sidebar.slider("Point display radius: ", 1, 10, 1)
    stroke_color = st.sidebar.color_picker("Stroke color: ")
    bg_color = st.sidebar.color_picker("Background color: ", "#AA9F9F")
    bg_image = st.sidebar.file_uploader("Choose controlling image:", type=["png", "jpg"])
    if bg_image is not None:
        file_bytes = np.asarray(bytearray(bg_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        cv2.imwrite("control.png", opencv_image)
    it_n = st.sidebar.number_input("Iterations number:", min_value=1, max_value=4)
    control = st.checkbox("Check if you want to add a controlling image")

    p = st.text_area("# Write your prompt here:", placeholder="The more specific you are, the better it is.")

    col2, col3 = st.columns([0.5, 0.5])

    with col2:
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=None,
            height=1000,
            width=1000,
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
            key="canvas",
        )

    new_images = []

    if canvas_result.image_data is not None:
        image = cv2.imwrite("output.png", canvas_result.image_data)

    with col3:
        col4, col5 = st.columns([0.75, 0.25])
        with col4:
            st.text("")
            st.text("")
            if st.button("START INFERENCE", type="primary"):
                with st.spinner("# Processing your image. Please wait"):
                    new_images = backend.inference(p, control, it_n, modelType, adapterType)
                    time.sleep(3)
                    if new_images == "Error":
                        st.write("# Something went wrong, check your inputs!")
                    else:
                        with zipfile.ZipFile("images.zip", "w", zipfile.ZIP_DEFLATED) as zipObj:
                            for i, new_image in enumerate(new_images):
                                new_image.save("image" + str(i) + ".jpg")
                                with open("image" + str(i) + ".jpg", "rb") as file:
                                    zipObj.write("./image" + str(i) + ".jpg")
                        with open("images.zip", "rb") as file:
                            btn = st.download_button(
                                label="Download your images",
                                data=file,
                                file_name="images.zip",
                                mime="application/zip"
                            )
        with col5:
            st.text("")
            st.text("")
            st.link_button("HUGGING FACE COMMUNITY", url="https://huggingface.co", type="primary")

elif appMode == "train":
    st.markdown("<h1 style='text-align: center; color: white;'>Welcome to SnapArt - Training Session</h1>",
                unsafe_allow_html=True)
    images = st.sidebar.file_uploader("Upload your images:", type=["png", "jpg"], accept_multiple_files=True)
    for i, image in enumerate(images):
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        cv2.imwrite("image" + str(i) + ".png".format(i=i), opencv_image)
    instancePrompt = st.text_area("# Write your instance prompt here:", placeholder="Example: a photo of a sks lamp")
    validationPrompt = st.text_area("# Write your validation prompt here:", placeholder="Example: a photo of a sks "
                                                                                        "lamp on a mountain")
    n = len(images)
    if st.button("START TRAINING", type="primary"):
        with st.spinner("# Processing your images (it might take a while...). Please wait"):
            safetensorsFile = backend.training(instancePrompt, validationPrompt, n)
            time.sleep(3)
        if safetensorsFile == "Error" or n == 0:
            st.write("# Something went wrong, check your inputs!")
        else:
            st.write("# Your LoRA file is ready for download! ")
            btn = st.download_button(
                label="Download",
                data=safetensorsFile,
                file_name="pytorch_lora_weights.safetensors".format(safetensorsFile),
                type="primary"
            )
