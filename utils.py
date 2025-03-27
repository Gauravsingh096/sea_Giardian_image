# # import torch
# # from torchvision import datasets, transforms
# # from torch.utils.data import DataLoader

# # def load_data(batch_size=32):
# #     transform = transforms.Compose([
# #         transforms.Resize((224, 224)),
# #         transforms.RandomHorizontalFlip(),
# #         transforms.RandomRotation(15),
# #         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
# #         transforms.ToTensor(),
# #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# #     ])

# #     dataset = datasets.ImageFolder(root="dataset/", transform=transform)
# # Replace utils.py transforms with this:
# import albumentations as A
# import torch
# from albumentations.pytorch import ToTensorV2
# from torchvision import datasets
# from torch.utils.data import DataLoader
# import numpy as np

# def load_data(batch_size=32):
#     transform = A.Compose([
#         A.Resize(224, 224),
#         A.HorizontalFlip(p=0.5),
#         A.Rotate(limit=15),
#         A.RandomBrightnessContrast(p=0.2),
#         A.CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=None, min_height=8, min_width=8, fill_value=0, p=0.3),
#         # A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),  # New
#         # A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, p=0.3),  # New
#         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ToTensorV2()
#     ])

#     dataset = datasets.ImageFolder(root="dataset/", transform=lambda x: transform(image=np.array(x))["image"])
#     train_size = int(0.8 * len(dataset))
#     test_size = len(dataset) - train_size
#     train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     return train_loader, test_loader
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np

# Define the transformation pipeline globally
transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15),
    A.RandomBrightnessContrast(p=0.2),
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=None, min_height=8, min_width=8, fill_value=0, p=0.3),
    # A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),  # Fixed
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def load_data(batch_size=32):
    """
    Load and preprocess the dataset for training and testing.

    Args:
        batch_size (int): Batch size for DataLoader.

    Returns:
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
    """
    # Load the dataset from the specified root directory
    dataset = datasets.ImageFolder(root="dataset/", transform=lambda x: transform(image=np.array(x))["image"])

    # Split the dataset into training and testing sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create DataLoader for training and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# updated pygame.examples.oldalien.main()