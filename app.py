# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import HTMLResponse
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import torch.nn as nn
# import torchvision.models as models
# import boto3
# import io
# import logging
# import uuid
# from asgiref.wsgi import WsgiToAsgi


# from dotenv import load_dotenv
# import os

# load_dotenv()  # Load .env variables

# AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
# AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
# AWS_REGION = os.getenv("AWS_REGION")
# S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")



# # Initialize FastAPI app
# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Logging setup
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# # -----------------------------
# # AWS S3 Configuration (update with your details)
# # -----------------------------
# S3_BUCKET = "sea-guardian-image"    # Your S3 bucket name
# S3_REGION = "us-east-1"              # Your AWS region

# s3_resource = boto3.resource(
#     "s3",
#     region_name=S3_REGION,
#     endpoint_url=f"https://s3.{S3_REGION}.amazonaws.com"
# )

# # -----------------------------
# # Model Configuration
# # -----------------------------
# MODEL_PATH = "models/model.pth"
# class_names = ["oil_spills", "plastic_waste", "illegal_dumping", "animals"]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Use ResNet50 for RGB images
# model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, len(class_names))

# # Load model weights from S3
# try:
#     obj = s3_resource.Object(S3_BUCKET, MODEL_PATH)
#     model_data = io.BytesIO()
#     obj.download_fileobj(model_data)
#     model_data.seek(0)
#     model.load_state_dict(torch.load(model_data, map_location=device), strict=False)
#     logging.info("✅ Model loaded from S3 successfully.")
# except Exception as e:
#     logging.error(f"❌ Error loading model from S3: {e}")
#     raise HTTPException(status_code=500, detail="Error loading model from S3")

# model = model.to(device)
# model.eval()

# # -----------------------------
# # Image Transformation (for RGB images)
# # -----------------------------
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# # -----------------------------
# # S3 Utility: Upload image to S3 (synchronous helper)
# # -----------------------------
# def upload_to_s3(contents: bytes) -> str:
#     filename = str(uuid.uuid4()) + ".jpg"
#     s3_client = boto3.client("s3", region_name=S3_REGION, endpoint_url=f"https://s3.{S3_REGION}.amazonaws.com")
#     s3_client.put_object(Bucket=S3_BUCKET, Key=filename, Body=contents, ContentType="image/jpeg")
#     image_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{filename}"
#     return image_url

# # -----------------------------
# # API Endpoints
# # -----------------------------

# @app.get("/", response_class=HTMLResponse)
# async def home():
#     html_content = """
#     <!DOCTYPE html>
#     <html lang="en">
#     <head>
#         <meta charset="UTF-8">
#         <title>Sea Guardian - Waste Classifier</title>
#     </head>
#     <body>
#         <h1>Upload an Image</h1>
#         <form id="uploadForm" enctype="multipart/form-data">
#             <input type="file" id="fileInput" accept="image/*" required>
#             <button type="submit">Predict</button>
#         </form>
    
#         <h2 id="result"></h2>
    
#         <script>
#             const form = document.getElementById('uploadForm');
#             form.addEventListener('submit', async (e) => {
#                 e.preventDefault();
#                 const fileInput = document.getElementById('fileInput');
#                 const formData = new FormData();
#                 formData.append('file', fileInput.files[0]);
    
#                 const response = await fetch('/predict', {\n                    method: 'POST',\n                    body: formData\n                });\n                \n                const data = await response.json();\n                document.getElementById('result').innerText = \"Predicted Class: \" + data.predicted_class;\n            });\n        </script>\n    </body>\n    </html>\n    """
#     return HTMLResponse(content=html_content)

# @app.post("/predict")
# async def predict_route(file: UploadFile = File(...)):
#     if not file:
#         raise HTTPException(status_code=400, detail="No file provided")
#     try:
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert("RGB")
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Error processing image: {e}")
    
#     try:
#         image = transform(image).unsqueeze(0).to(device)
#         with torch.no_grad():
#             outputs = model(image)
#             _, predicted = torch.max(outputs, 1)
#         predicted_label = class_names[predicted.item()]
#         logging.info(f"Predicted class: {predicted_label}")
#         return {"predicted_class": predicted_label}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

# @app.post("/upload")
# async def upload(file: UploadFile = File(...)):
#     if not file:
#         raise HTTPException(status_code=400, detail="No file provided")
#     try:
#         contents = await file.read()
#         image_url = upload_to_s3(contents)
#         return {"image_url": image_url}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error uploading image: {e}")

# # -----------------------------
# # Run the Server using Uvicorn
# # -----------------------------
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import boto3
import io
import logging
import uuid
import os
import gc

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -----------------------------
# AWS S3 Configuration
# -----------------------------
S3_BUCKET = "sea-guardian-image"
S3_REGION = "us-east-1"

s3_resource = boto3.resource(
    "s3",
    region_name=S3_REGION,
    endpoint_url=f"https://s3.{S3_REGION}.amazonaws.com"
)

# -----------------------------
# Model Configuration (Replaced ResNet50 with MobileNetV2)
# -----------------------------
MODEL_PATH = "models/model.pth"
class_names = ["oil_spills", "plastic_waste", "illegal_dumping", "animals"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load MobileNetV2 Instead of ResNet50
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Modify the classifier for 4 classes
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(class_names))

# Load model weights from S3
try:
    obj = s3_resource.Object(S3_BUCKET, MODEL_PATH)
    model_data = io.BytesIO()
    obj.download_fileobj(model_data)
    model_data.seek(0)
    
    model.load_state_dict(torch.load(model_data, map_location=device), strict=False)
    logging.info("✅ Model loaded from S3 successfully.")
except Exception as e:
    logging.error(f"❌ Error loading model from S3: {e}")
    raise HTTPException(status_code=500, detail="Error loading model from S3")

model = model.to(device)
model.eval()

# -----------------------------
# Image Transformation
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

# -----------------------------
# S3 Utility: Upload Image to S3
# -----------------------------
def upload_to_s3(contents: bytes) -> str:
    filename = str(uuid.uuid4()) + ".jpg"
    s3_client = boto3.client("s3", region_name=S3_REGION, endpoint_url=f"https://s3.{S3_REGION}.amazonaws.com")
    s3_client.put_object(Bucket=S3_BUCKET, Key=filename, Body=contents, ContentType="image/jpeg")
    image_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{filename}"
    return image_url

# -----------------------------
# API Endpoints
# -----------------------------

@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Sea Guardian - Waste Classifier</title>
    </head>
    <body>
        <h1>Upload an Image</h1>
        <form id="uploadForm" enctype="multipart/form-data">
            <input type="file" id="fileInput" accept="image/*" required>
            <button type="submit">Predict</button>
        </form>
    
        <h2 id="result"></h2>
    
        <script>
            const form = document.getElementById('uploadForm');
            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                const fileInput = document.getElementById('fileInput');
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
    
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                document.getElementById('result').innerText = "Predicted Class: " + data.predicted_class;
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")
    
    try:
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
        predicted_label = class_names[predicted.item()]
        logging.info(f"Predicted class: {predicted_label}")

        # ✅ Free memory
        del image
        gc.collect()
        torch.cuda.empty_cache()

        return {"predicted_class": predicted_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    try:
        contents = await file.read()
        image_url = upload_to_s3(contents)
        return {"image_url": image_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading image: {e}")

# -----------------------------
# Run the Server using Uvicorn
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
