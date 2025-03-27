from train import train_model
from test import evaluate_model
import torch

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Train the model
train_model()

# Evaluate model
evaluate_model()
# updated one