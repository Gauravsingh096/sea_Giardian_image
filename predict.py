# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import torch.nn as nn
# import torchvision.models as models
# from torchvision.models import ResNet50_Weights
# from utils import transform  # Import the transform from utils.py

# class_names = ["animals", "illegal_dumping", "oil_spill", "plastic_waste"]
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Ensure reproducibility
# torch.manual_seed(0)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(0)

# def predict_image(image_path):
#     weights = ResNet50_Weights.IMAGENET1K_V1
#     model = models.resnet50(weights=weights)
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Linear(num_ftrs, len(class_names))

#     try:
#         model.load_state_dict(torch.load("models/model.pth", map_location=device))
#     except FileNotFoundError:
#         print("Error: Model file not found.")
#         return
#     except Exception as e:
#         print(f"Error loading the model: {e}")
#         return

#     model = model.to(device)
#     model.eval()  # Set the model to evaluation mode

#     try:
#         image = Image.open(image_path).convert("RGB")  # Convert to RGB
#         image = transform(image=np.array(image))["image"]  # Apply the transform
#         image = torch.tensor(image).unsqueeze(0).to(device)  # Convert to tensor and add batch dimension
#     except FileNotFoundError:
#         print("Error: Image file not found.")
#         return
#     except Exception as e:
#         print(f"Error processing the image: {e}")
#         return

#     with torch.no_grad():
#         outputs = model(image)
#         _, predicted = torch.max(outputs, 1)

#     print(f"Predicted class: {class_names[predicted.item()]}")

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) > 1:
#         image_path = sys.argv[1]
#     else:
#         image_path = "oil.png"  # Default image path

#     predict_image(image_path)

from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Class labels
class_names = ["oil_spills", "plastic_waste", "illegal_dumping", "animals"]

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)  # Do not load ImageNet weights
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load("models/model.pth", map_location=device))
model = model.to(device)
model.eval()

# Define transformation for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

@app.route('/')
def home():
    return "Sea Guardian API is running. Use /predict to classify images.", 200

@app.route('/predict', methods=['POST'])
def predict_route():
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Open the uploaded image file
        image = Image.open(file).convert('RGB')
    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 400

    # Preprocess the image
    image = transform(image).unsqueeze(0).to(device)

    # Run prediction using the loaded model
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    predicted_label = class_names[predicted.item()]

    # Return the predicted label as JSON
    return jsonify({"predicted_class": predicted_label}), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
# updated one