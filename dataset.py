import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

def get_dataloader(dataset_name="CIFAR100", batch_size=32, num_workers=4, data_dir="./data"):
    """
    Get the DataLoader for CIFAR-10 or CIFAR-100 dataset.

    Args:
        dataset_name (str): "CIFAR10" or "CIFAR100"
        batch_size (int): Number of samples per batch
        num_workers (int): Number of worker threads for data loading
        data_dir (str): Directory to store the dataset

    Returns:
        train_loader (DataLoader): DataLoader for the training set
        test_loader (DataLoader): DataLoader for the test set
    """

    # Select dataset and set normalization parameters
    if dataset_name.upper() == "CIFAR100":
        dataset_class = torchvision.datasets.CIFAR100
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif dataset_name.upper() == "CIFAR10":
        dataset_class = torchvision.datasets.CIFAR10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else:
        raise ValueError("dataset_name must be 'CIFAR10' or 'CIFAR100'")

    # Data augmentation for the training set
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Transformation for the test set (without augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load dataset
    train_dataset = dataset_class(root=data_dir, train=True, transform=transform_train, download=True)
    test_dataset = dataset_class(root=data_dir, train=False, transform=transform_test, download=True)

    # Create DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
