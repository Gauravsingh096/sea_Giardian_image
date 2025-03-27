import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from utils import load_data
from torchvision.models import ResNet50_Weights
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_model(epochs=15):
    train_loader, _ = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)  # Adjust for 4 classes
    model = model.to(device)

    # Define Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Training Loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()  
            scheduler.step()  

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")
        print("Gauravsingh096")

    # Save model
    torch.save(model.state_dict(), "models/model.pth")
    print("Model saved to models/model.pth")