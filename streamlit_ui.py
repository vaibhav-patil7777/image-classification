# ===============================
# Advanced Image Classification App
# PyTorch + Streamlit
# ===============================

import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# -------------------------------
# Page Config (Responsive)
# -------------------------------
st.set_page_config(
    page_title="AI Image Classifier",
    page_icon="🧠",
    layout="centered"
)

# -------------------------------
# Custom CSS (Modern UI)
# -------------------------------
st.markdown("""
<style>
.main {
    padding: 1rem;
}
.title {
    text-align: center;
    font-size: 2.2rem;
    font-weight: 700;
}
.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 20px;
}
.upload-box {
    border: 2px dashed #4CAF50;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Title Section
# -------------------------------
st.markdown('<div class="title">🧠 AI Image Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload any image and get Top-5 predictions instantly</div>', unsafe_allow_html=True)

# -------------------------------
# Load Model (Cached)
# -------------------------------
@st.cache_resource
def load_model():
    model = models.resnet101(pretrained=True)
    model.eval()
    return model

model = load_model()

# -------------------------------
# Image Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# Reset Function
# -------------------------------
def reset_app():
    st.session_state.clear()

# -------------------------------
# Upload Section
# -------------------------------
st.markdown('<div class="upload-box">📂 Upload Image</div>', unsafe_allow_html=True)

file_up = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"]
)

# -------------------------------
# Prediction Function
# -------------------------------
def predict(image):
    img = Image.open(image).convert("RGB")
    batch = torch.unsqueeze(transform(img), 0)

    with torch.no_grad():
        outputs = model(batch)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0] * 100

    with open("imagenet_classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]

    top5 = torch.topk(probs, 5)
    results = []

    for i in range(5):
        results.append({
            "Class": classes[top5.indices[i]],
            "Confidence (%)": round(top5.values[i].item(), 2)
        })

    return img, pd.DataFrame(results)

# -------------------------------
# Display Results
# -------------------------------
if file_up is not None:
    st.write("")

    img, df = predict(file_up)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    st.subheader("🔍 Top-5 Predictions")
    st.table(df)

    st.success("Prediction completed successfully!")

    # Refresh Button
    st.button("🔄 Upload New Image", on_click=reset_app)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown(
    "<center>🚀 Built with PyTorch & Streamlit</center>",
    unsafe_allow_html=True
)
