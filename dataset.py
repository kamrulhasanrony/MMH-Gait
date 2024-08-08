import os
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import datasets, transforms

def collate_fn(batch):
    """
    Custom collate function to resize images and stack them into a single tensor.
    Args:
        batch (list of tuples): A batch of samples, where each sample is a tuple (image, label).
                                - image (PIL Image or Tensor): The input image.
                                - label (int or Tensor): The corresponding label.
    Returns:
        tuple: A tuple containing:
               - images_tensor (Tensor): A tensor of resized images of shape (batch_size, 3, 224, 224).
               - labels_tensor (Tensor): A tensor of labels of shape (batch_size,).
    """
    fixed_size = (224, 224)
    resized_images = []
    labels = []
    for sample in batch:
        image, label = sample
        resized_image = transforms.functional.resize(image, fixed_size)
        resized_images.append(resized_image)
        labels.append(label)
    images_tensor = torch.stack(resized_images)
    labels_tensor = torch.tensor(labels)
    return images_tensor, labels_tensor

def data_divider(root_path, batch_size, model_type):
    """
    Loads and prepares training and testing data loaders with specified batch size and transformations.
    Args:
        root_path (str): Root directory containing 'train' and 'test' subdirectories.
        batch_size (int): Number of samples per batch to load.
        model_type (str): Type of the model (which model is used . E.g., resnet50, vgg16 etc).
    Returns:
        tuple: A tuple containing:
               - train_loader (DataLoader): DataLoader for the training dataset.
               - test_loader (DataLoader): DataLoader for the testing dataset.
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    size = 224
    train_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_path = os.path.join(root_path, 'train')
    test_path = os.path.join(root_path, 'test')
    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=collate_fn)
    print('Datasets Loaded Successfully!')
    return train_loader, test_loader
