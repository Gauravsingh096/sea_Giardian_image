# from flask import Flask, request, jsonify, render_template
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import torch.nn as nn
# import torchvision.models as models
# import os
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app) 
# # Class labels
# class_names = ["oil_spills", "plastic_waste", "illegal_dumping", "animals"]

# # Load model once
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = models.resnet50(pretrained=True)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, len(class_names))
# model.load_state_dict(torch.load("models/model.pth", map_location=device))
# model = model.to(device)
# model.eval()

# # Prediction function
# def predict(image_path):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5])
#     ])
#     image = Image.open(image_path).convert('RGB')
#     image = transform(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         outputs = model(image)
#         _, predicted = torch.max(outputs, 1)

#     return class_names[predicted.item()]

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict_api():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files['file']
#     file_path = os.path.join("uploads", file.filename)
#     os.makedirs("uploads", exist_ok=True)
#     file.save(file_path)

#     result = predict(file_path)
#     os.remove(file_path)

#     return jsonify({"prediction": result})

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import os
from flask_cors import CORS
import os
import torchvision.models as models
model = models.resnet18()

app = Flask(__name__)
CORS(app)


model_path = "models/model.pth"

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    print("Model loaded successfully.")
else:
    print(f"Error: Model file '{model_path}' not found!")
    
model.eval()

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
    # Check if a file is provided in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Open the image and convert it to RGB
        image = Image.open(file).convert('RGB')
    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 400

    # Preprocess the image
    image = transform(image).unsqueeze(0).to(device)

    # Run the model and get prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    predicted_label = class_names[predicted.item()]
    return jsonify({"predicted_class": predicted_label}), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
# updated one