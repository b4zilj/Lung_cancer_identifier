import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import mlflow.pytorch


MODEL_URI = "runs:/923330918ae841f48239d3bb7b8df35e/model"  # Replace <RUN_ID> with your MLflow run ID
CLASS_NAMES = ["cancer", "normal"]
IMG_SIZE = 224


@st.cache_resource
def load_model():
    model = mlflow.pytorch.load_model(MODEL_URI)
    model.eval()
    return model

model = load_model()


def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


st.title("🫁 Lung Cancer Detector")
st.write("Upload an image to check if it's *cancer* or *normal*.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    
    input_tensor = transform_image(image)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        label = CLASS_NAMES[predicted.item()]

    st.markdown(f"### 🩺 Prediction: *{label.upper()}*")