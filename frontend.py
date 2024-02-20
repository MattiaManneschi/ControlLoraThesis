import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import backend
from io import BytesIO
import imageio.v3 as iio

buf = BytesIO()

drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
)

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
it_n = st.sidebar.number_input("Iterations number:", min_value=1, max_value=10)

realtime_update = st.sidebar.checkbox("Update in realtime", True)
fast_draw = st.sidebar.checkbox("Use background image", False)

p = st.text_area("# Write your positive prompt here:")
n_p = st.text_area("# Write your negative prompt here:")

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=300,
    drawing_mode=drawing_mode,
    point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
    key="canvas",
)

new_images = []

if st.sidebar.button("Continue"):
    if fast_draw is True:
        new_images = backend.scribbleInf(iio.imread(bg_image), p, n_p, it_n)
    else:
        new_images = backend.scribbleInf(canvas_result.image_data, p, n_p, it_n)
    for new_image in new_images:
        st.image(new_image)
        new_image.save(buf, format="PNG")
        byte_image = buf.getvalue()
        btn = st.download_button(
            label="Download image",
            data=byte_image,
            file_name="imagename{}.png".format(new_image),
            mime="image/png"
        )

if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)
if canvas_result.json_data is not None:
    objects = pd.json_normalize(canvas_result.json_data["objects"])
    for col in objects.select_dtypes(include=['object']).columns:
        objects[col] = objects[col].astype("str")
    st.dataframe(objects)
