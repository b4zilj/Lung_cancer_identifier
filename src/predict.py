import mlflow.pytorch
import torch
from torchvision import transforms
from PIL import Image

MODEL_URI = "runs:/923330918ae841f48239d3bb7b8df35e/model" 
IMG_SIZE = 224
CLASS_NAMES = ["cancer", "normal"]


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def predict(image_path):
    
    model = mlflow.pytorch.load_model(MODEL_URI)
    model.eval()


    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  

    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_name = CLASS_NAMES[predicted.item()]
    
    print(f"Prediction: {class_name}")
    return class_name


if __name__ == "__main__":
    img_path = "data/raw/normal/1.png" 
    predict(img_path)