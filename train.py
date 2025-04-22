import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

# model
from models.fc_model import FCModel
from models.cnn_model import CNNModel
from models.transformer_model import TransformerModel

# hyperparams
from config.hyperparams import *
from config.hyperparams import get_hyperparams


# ---------------- log config ---------------------
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, 'train.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)-s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )


# ---------------- plot functin -------------------
def plot_figure(results_dir, train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'loss_accuracy_curves.png')
    plt.savefig(plot_path)
    logging.info(f"Training curves saved to {plot_path}")
    plt.close()

# ------------------- train -----------------------
def train(args, hyperparams):
    # device & log
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    #  file directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('runs', f"{args.model_type}_{timestamp}")
    log_dir = os.path.join(run_dir, 'logs')
    results_dir = os.path.join(run_dir, 'results')
    model_save_path = os.path.join(run_dir, f'{args.model_type}_best.pth')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Tensorboard
    writer = SummaryWriter(log_dir=log_dir)

    # loading data & processing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 标准化参数
    ])

    # get data
    try:
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    except Exception as e:
        logging.error(f"Error downloading/loading MNIST dataset: {e}")
        logging.error("Please check your internet connection or dataset path.")
        return

    # split train & val
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=hyperparams['batch_size'], shuffle=True,num_workers=args.num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=hyperparams['batch_size'], shuffle=False,num_workers=args.num_workers)
    # 为了考虑日志完整性，我选择加上测试数据
    test_loader = DataLoader(dataset=test_dataset, batch_size=hyperparams['batch_size'], shuffle=False,num_workers=args.num_workers)
    logging.info(f"Dataset loaded: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}")

    # model
    if args.model_type == 'fc':
        # FC模型可能需要hidden_sizes，从hyperparams获取
        model = FCModel(num_classes=hyperparams['num_classes'],
                        hidden_size=hyperparams.get('hidden_sizes'),
                        dropout=hyperparams['dropout_rate']).to(device)
    elif args.model_type == 'cnn':
        model = CNNModel(num_classes=hyperparams['num_classes'], dropout=hyperparams['dropout_rate']).to(device)
    elif args.model_type == 'transformer':
        model = TransformerModel(num_classes=hyperparams['num_classes'],
                                 dmodel=hyperparams['d_model'],
                                 head=hyperparams['nhead'],
                                 num_layers=hyperparams['num_layers'],
                                 dim_ffn=hyperparams['dim_feedforward'],
                                 dropout=hyperparams['dropout_rate']).to(device)
    else:
        logging.error(f"Unsupported model type: {args.model_type}")
        return
    logging.info(f"Model selected: {args.model_type}")
    logging.info(model)

    # Loss Function & Optimizer
    criterion = nn.CrossEntropyLoss()
    if hyperparams['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'], weight_decay=hyperparams.get('weight_decay', 0))
    else:
        logging.warning(f"Unsupported optimizer type: {hyperparams['optimizer']}. Using Adam as default.")
        optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'], weight_decay=hyperparams.get('weight_decay', 0))
    logging.info(f"Optimizer: {hyperparams['optimizer']}, LR: {hyperparams['learning_rate']}, Weight Decay: {hyperparams.get('weight_decay', 0)}")

    # Learning Rate Scheduler
    scheduler = None
    scheduler_type = hyperparams.get('scheduler')
    if scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=hyperparams['step_size'], gamma=hyperparams['gamma'])
        logging.info(f"Using StepLR scheduler with step_size={hyperparams['step_size']}, gamma={hyperparams['gamma']}")
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=hyperparams['T_max'])
        logging.info(f"Using CosineAnnealingLR scheduler with T_max={hyperparams['T_max']}")
    else:
        logging.info("No learning rate scheduler specified or scheduler type not supported.")


    # train loop
    best_val_accuracy = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    logging.info("Starting training...")
    for epoch in range(hyperparams['epochs']):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # tqdm
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{hyperparams['epochs']} [Train]")
        for i, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()


            batch_loss = loss.item()
            batch_acc = (predicted == labels).sum().item() / labels.size(0)
            train_pbar.set_postfix({'Loss': f'{batch_loss:.4f}', 'Acc': f'{batch_acc:.4f}'})


            if (i + 1) % hyperparams['log_interval'] == 0:
                # 日志记录
                logging.info(f'Epoch [{epoch+1}/{hyperparams["epochs"]}], Step [{i+1}/{len(train_loader)}], Loss: {batch_loss:.4f}, Acc: {batch_acc:.4f}')
                # TensorBoard
                writer.add_scalar('Training/Batch_Loss', batch_loss, epoch * len(train_loader) + i)
                writer.add_scalar('Training/Batch_Accuracy', batch_acc, epoch * len(train_loader) + i)

        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        writer.add_scalar('Training/Epoch_Loss', epoch_train_loss, epoch + 1)
        writer.add_scalar('Training/Epoch_Accuracy', epoch_train_acc, epoch + 1)

        # --- val ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{hyperparams['epochs']} [Val]")
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{(predicted == labels).sum().item() / labels.size(0):.4f}'})

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        writer.add_scalar('Validation/Epoch_Loss', epoch_val_loss, epoch + 1)
        writer.add_scalar('Validation/Epoch_Accuracy', epoch_val_acc, epoch + 1)

        logging.info(
            f'Epoch [{epoch + 1}/{hyperparams["epochs"]}] Summary: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')

        # --- Svae best model ---
        if epoch_val_acc > best_val_accuracy:
            best_val_accuracy = epoch_val_acc
            torch.save(model.state_dict(), model_save_path)
            logging.info(
                f"Best model saved to {model_save_path} with validation accuracy: {best_val_accuracy:.4f}")

        # --- Learning Rate Scheduler Step ---
        if scheduler:
            scheduler.step()
            # Optionally log the learning rate
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Training/Learning_Rate', current_lr, epoch + 1)
            logging.info(f"Epoch {epoch + 1}: Learning rate updated to {current_lr:.6f}")


    logging.info("Training finished.")
    writer.close()

    # --- 绘制并保存曲线 ---
    plot_figure(results_dir, train_losses, val_losses, train_accs, val_accs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
    parser.add_argument('--model-type', type=str, default='cnn', choices=['fc', 'cnn', 'transformer'],
                        help='Type of model to train (fc, cnn, transformer)')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='number of data loading workers (default: 2)')
    args = parser.parse_args()

    try:
        hyperparams = get_hyperparams(args.model_type)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    # 设置日志记录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('runs', f"{hyperparams['model_name']}_{timestamp}")
    log_dir = os.path.join(run_dir, 'logs')
    setup_logging(log_dir)

    logging.info(f"Arguments: {args}")
    logging.info(f"Hyperparameters for {args.model_type}: {hyperparams}")

    train(args, hyperparams)






