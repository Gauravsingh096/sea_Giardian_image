import torch
from utils import load_data
import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from sklearn.metrics import classification_report

class_names = ["animals", "illegal_dumping", "oil_spill", "plastic_waste"]

def evaluate_model():
    _, test_loader = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)  # Adjust for 4 classes
    model.load_state_dict(torch.load("models/model.pth", map_location=device))
    model = model.to(device)
    model.eval()

    correct, total = 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Ensure 'predicted' is defined here
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    print(classification_report(all_labels, all_preds, target_names=class_names))