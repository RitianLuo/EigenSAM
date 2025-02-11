import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import random
import numpy as np

from sam import SAM
from resnet import get_resnet
from dataset import get_dataloader
from utils import estimate_largest_eigenvector, modify_gradient_with_projection
from bypass_bn import enable_running_stats, disable_running_stats


def parse_args():
    """
    Parse command-line arguments using argparse.
    You can adjust or split this function further if you have more complex requirements.
    """
    parser = argparse.ArgumentParser(description="Train ResNet on CIFAR-10/100 with SAM Optimizer")
    parser.add_argument("--resnet_version", type=int, choices=[18, 34, 50, 101, 152], default=18,
                        help="Specify which ResNet version to use (18, 34, 50, 101, 152)")
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["CIFAR10", "CIFAR100"], help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--epochs", type=int, default=10, help="Total number of training epochs")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay for optimizer")
    parser.add_argument("--rho", type=float, default=0.05, help="Rho parameter for SAM")
    parser.add_argument("--sam", type=str, default="SGD", help="Which SAM mode to use: [SGD, SAM, Eigen-SAM]")
    parser.add_argument("--alpha", type=float, default=0.2, help="Alpha parameter for Eigen-SAM")
    parser.add_argument("--freq", type=int, default=100, help="Frequency for updating the largest eigenvector")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--log_file", type=str, default="./log/", help="File path prefix for saving training logs")
    args = parser.parse_args()
    return args


def train(model, train_loader, test_loader, optimizer, criterion, device, epochs, args, scheduler, log_file):
    """
    Main training function.
    
    Args:
        model: The neural network model to train.
        train_loader: DataLoader for the training dataset.
        test_loader: DataLoader for the test/validation dataset.
        optimizer: The optimizer (could be SAM, Eigen-SAM, or standard SGD).
        criterion: Loss function (e.g., CrossEntropyLoss).
        device: The torch device (CPU or GPU) on which to train.
        epochs: Number of training epochs.
        args: Parsed command-line arguments (including hyperparameters).
        scheduler: Learning rate scheduler (e.g., CosineAnnealingLR).
        log_file: A file handle for logging (opened in main).
    """
    if args.sam == "Eigen-SAM":
        freq = args.freq
        # Will store the top eigenvector v
        v = None 

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            if args.sam == 'SAM':
                # --- Standard SAM mode ---
                optimizer.zero_grad()
                enable_running_stats(model)
                logits = model(images)
                loss = criterion(logits, labels)
                total_loss += loss.item()

                loss.backward()
                optimizer.first_step(zero_grad=True)

                disable_running_stats(model)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.second_step(zero_grad=True)

            elif args.sam == 'Eigen-SAM':
                # ============ Eigen-SAM logic ============
                # 1) Periodically update the largest eigenvector v
                if step % freq == 0 and step > 0:
                    v = estimate_largest_eigenvector(
                        model, criterion, v, images, labels, steps=5
                    )

                # 2) First forward & backward pass
                optimizer.zero_grad()
                enable_running_stats(model)
                logits = model(images)
                loss = criterion(logits, labels)
                total_loss += loss.item()
                loss.backward()

                # 3) Modify gradients using the stored eigenvector v
                if v is not None:
                    modify_gradient_with_projection(model, v, alpha=args.alpha)

                # 4) First step of SAM
                optimizer.first_step(zero_grad=True, normalization=False)

                # 5) Second forward & backward pass
                disable_running_stats(model)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()

                # 6) Second step of SAM
                optimizer.second_step(zero_grad=True)

            else:
                # --- Plain SGD ---
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

        train_loss = total_loss / len(train_loader)

        # Validation using the provided test_loader
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Accuracy: {test_acc:.2f}%"
        )
        log_file.write(f"{epoch+1}, {train_loss:.4f}, {test_loss:.4f}, {test_acc:.2f}\n")
        log_file.flush()

        scheduler.step()


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the model on a given dataset.
    
    Args:
        model (torch.nn.Module): The neural network model to be evaluated.
        test_loader (torch.utils.data.DataLoader): DataLoader providing the test/validation dataset.
        criterion (torch.nn.Module): Loss function (e.g., CrossEntropyLoss).
        device (torch.device): The device on which to run the evaluation (CPU or GPU).
    
    Returns:
        test_loss (float): The average loss over the entire test dataset.
        test_acc (float): The test accuracy (in percentage).
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)  # Forward pass
            loss = criterion(logits, labels)  # Compute loss
            total_loss += loss.item()

            _, predicted = logits.max(1)  # Get the index of the max log-probability
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    # Compute average loss and overall accuracy
    test_loss = total_loss / len(test_loader)
    test_acc = 100.0 * correct / total
    return test_loss, test_acc


def main():
    """
    Main entry point for the script. This function:
    1) Parses command-line arguments.
    2) Sets the random seed.
    3) Prepares data loaders (train_loader, test_loader).
    4) Initializes the model, optimizer, and scheduler.
    5) Opens the log file.
    6) Calls the training function.
    7) Closes the log file.
    """
    args = parse_args()

    # Set random seed for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Build the filename for logging (include various hyperparameters)
    param_list = []
    for key, value in vars(args).items():
        if key == "log_file":
            continue
        param_list.append(f"{key}-{value}")
    param_str = "_".join(param_list)
    args.log_file += f"training_{param_str}.txt"

    # Select the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data loaders. IMPORTANT: capture both train_loader and test_loader
    train_loader, test_loader = get_dataloader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    num_classes = 100 if args.dataset == "CIFAR100" else 10

    # Create the model
    model = get_resnet(args.resnet_version, num_classes=num_classes).to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Choose the optimizer / SAM variant
    if args.sam == "SAM":
        # Standard SAM
        base_optimizer = torch.optim.SGD
        optimizer = SAM(
            model.parameters(),
            base_optimizer,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            rho=args.rho
        )
    elif args.sam == "Eigen-SAM":
        # Eigen-SAM
        base_optimizer = torch.optim.SGD
        optimizer = SAM(
            model.parameters(),
            base_optimizer,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            rho=args.rho
        )
    else:
        # Plain SGD
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Open the log file
    log_file = open(args.log_file, "w")
    log_file.write("Epoch, Training Loss, Test Loss, Test Accuracy\n")

    # Start training, now passing test_loader as well
    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        args=args,
        scheduler=scheduler,
        log_file=log_file
    )

    # Close the log file
    log_file.close()


# Script entry point
if __name__ == "__main__":
    main()
