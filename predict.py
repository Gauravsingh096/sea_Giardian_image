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
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from utils import transform  # Import the transform from utils.py
import numpy as np  # Import numpy

class_names = ["animals", "illegal_dumping", "oil_spill", "plastic_waste"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure reproducibility
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)

def predict_image(image_path):
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    try:
        model.load_state_dict(torch.load("models/model.pth", map_location=device))
    except FileNotFoundError:
        print("Error: Model file not found.")
        return
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    try:
        image = Image.open(image_path).convert("RGB")  # Convert to RGB
        image = transform(image=np.array(image))["image"]  # Apply the transform
        image = torch.tensor(image).unsqueeze(0).to(device)
        # image = torch.from_numpy(image).unsqueeze(0).to(device)
    except FileNotFoundError:
        print("Error: Image file not found.")
        return
    except Exception as e:
        print(f"Error processing the image: {e}")
        return

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    print(f"Predicted class: {class_names[predicted.item()]}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "test.png"  # Default image path

    predict_image(image_path)