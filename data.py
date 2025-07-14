from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import *

def get_transforms(model_name):
    size = 299 if model_name in ['inception_v3', 'xception'] else 224
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

def get_dataloaders(model_name):
    transform = get_transforms(model_name)
    
    train_set = datasets.ImageFolder(TRAIN_PATH, transform)
    val_set = datasets.ImageFolder(VAL_PATH, transform)
    test_set = datasets.ImageFolder(TEST_PATH, transform)
    
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, BATCH_SIZE)
    test_loader = DataLoader(test_set, BATCH_SIZE)
    
    return train_loader, val_loader, test_loader, train_set.classes