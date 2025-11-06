import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch

# ---------------------------------------
# 🎯 Load model
# ---------------------------------------
MODEL_PATH = "runs_skin_cancer/yolov8_skin_cls/weights/best.pt"
model = YOLO(MODEL_PATH)

st.title("🩺 Skin Cancer Detection using YOLOv8")
st.write("Upload a skin lesion image to predict the type of skin cancer.")

# ---------------------------------------
# 📤 File uploader
# ---------------------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run prediction
    st.write("🔍 Running prediction...")
    results = model.predict(image, imgsz=640, device=0 if torch.cuda.is_available() else "cpu")

    # Parse prediction
    pred_class = results[0].names[results[0].probs.top1]
    confidence = results[0].probs.top1conf.item() * 100

    # Display results
    st.success(f"### 🧠 Prediction: **{pred_class}**")
    st.write(f"Confidence: **{confidence:.2f}%**")
