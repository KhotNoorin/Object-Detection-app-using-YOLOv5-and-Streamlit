import streamlit as st
import torch
from PIL import Image
import numpy as np
import os
from datetime import datetime

st.set_page_config(page_title="YOLOv5 Object Detection", layout="wide")
st.title("YOLOv5 Object Detection App")

@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    return model

model = load_model()

st.sidebar.title("Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
model.conf = confidence

st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img_np = np.array(image)

    st.markdown("## Detection Results")
    with st.spinner("Running object detection..."):
        results = model(img_np)

    results.render()
    detected_img = Image.fromarray(results.ims[0])
    st.image(detected_img, caption="Detected Image", use_column_width=True)

    detections = results.pandas().xyxy[0]
    if not detections.empty:
        st.markdown("### Detection Table")
        st.dataframe(detections[['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']])
    else:
        st.warning("No objects detected with current threshold.")

    save_dir = "outputs"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"detection_{timestamp}.jpg")
    detected_img.save(save_path)
    st.success(f"Image saved to {save_path}")

else:
    st.info("Upload an image to start detection.")