import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import os
import logging
import matplotlib.pyplot as plt
colors = plt.get_cmap('viridis')
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
from datetime import datetime


# models
from models.fc_model import FCModel
from models.cnn_model import CNNModel
from models.transformer_model import TransformerModel

# log config
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, 'test.log')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_filename),
                            logging.StreamHandler()
                        ])


# Confusion Matrix
def plot_confusion_matrix(results_dir, y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plot_path = os.path.join(results_dir, 'confusion_matrix.png')
    plt.savefig(plot_path)
    logging.info(f"Confusion matrix saved to {plot_path}")
    plt.close()


# Precision-Recall
def plot_precision_recall_curve(results_dir, y_true, y_score, classes):
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    # 为每个类别计算PR曲线和AP
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(y_true_bin[:, i], y_score[:, i])

    # 绘制每个类别的PR曲线
    plt.figure(figsize=(10, 8))
    colors = plt.get_cmap('viridis')
    for i, color in zip(range(n_classes), colors(np.linspace(0, 1, n_classes))):
        plt.plot(recall[i], precision[i], color=color,
                 label=f'PR curve of class {classes[i]} (AP = {average_precision[i]:0.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve for each class')
    plt.legend(loc="best")
    plot_path = os.path.join(results_dir, 'precision_recall_curves.png')
    plt.savefig(plot_path)
    logging.info(f"Precision-Recall curves saved to {plot_path}")
    plt.close()


# ROC
def plot_roc_curve(results_dir, y_true, y_score, classes):
    n_classes = len(classes)
    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    # 为每个类别计算ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 绘制每个类别的ROC曲线
    plt.figure(figsize=(10, 8))
    colors = plt.get_cmap('viridis')
    for i, color in zip(range(n_classes), colors(np.linspace(0, 1, n_classes))):
        plt.plot(fpr[i], tpr[i], color=color,
                 label=f'ROC curve of class {classes[i]} (AUC = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) curve for each class')
    plt.legend(loc="lower right")
    plot_path = os.path.join(results_dir, 'roc_curves.png')
    plt.savefig(plot_path)
    logging.info(f"ROC curves saved to {plot_path}")
    plt.close()

# test
def test(args, results_dir): # Accept results_dir as an argument
    # -----device-----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Directory validation (model path existence checked in main)
    logging.info(f"Using results directory: {results_dir}")
    # No need to find results_dir from handlers anymore

    # loading data & processing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    try:
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    except Exception as e:
        logging.error(f"Error downloading/loading MNIST test dataset: {e}")
        return

    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    logging.info(f"Test dataset loaded: {len(test_dataset)} samples")

    # loading model
    if args.model_type == 'fc':
        model = FCModel(num_classes=10).to(device) # Dropout在评估时通常不启用，但加载权重时结构需匹配
    elif args.model_type == 'cnn':
        model = CNNModel(num_classes=10).to(device)
    elif args.model_type == 'transformer':
         model = TransformerModel(num_classes=10).to(device)
    else:
        logging.error(f"Unsupported model type: {args.model_type}")
        return

    model.eval()
    # 加载训练好的权重
    if args.model_path:
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)

    # val loop
    all_labels = []
    all_preds = []
    all_scores = []
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    logging.info("Starting evaluation...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)

            scores = torch.softmax(outputs, dim=1) # 获取概率分数
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_scores.append(scores.cpu().numpy()) # 保存分数

    all_scores = np.concatenate(all_scores, axis=0) # 合并分数

    # 计算指标
    avg_test_loss = test_loss / len(test_dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(10)], zero_division=0)

    logging.info(f"\n--- Test Results ---")
    logging.info(f"Average Test Loss: {avg_test_loss:.4f}")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"\nClassification Report:\n{report}")




    # 指标绘制
    plot_confusion_matrix(results_dir, all_labels, all_preds, classes=[str(i) for i in range(10)])
    plot_precision_recall_curve(results_dir, all_labels, all_scores, classes=[str(i) for i in range(10)])
    plot_roc_curve(results_dir, all_labels, all_scores, classes=[str(i) for i in range(10)])

    logging.info("Testing finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Testing')
    parser.add_argument('--model-type', type=str, required=True, choices=['fc', 'cnn', 'transformer'],
                        help='Type of model to test (fc, cnn, transformer)')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained model weights (.pth file)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='number of data loading workers (default: 2)')

    args = parser.parse_args()

    # --- Setup Logging Based on Args ---
    log_dir = None
    if args.model_path and os.path.exists(os.path.dirname(args.model_path)):
        run_dir = os.path.dirname(args.model_path)
        results_dir = os.path.join(run_dir, 'test_results')
        log_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        setup_logging(log_dir)
        logging.info(f"Arguments: {args}")
        logging.info(f"Logging and results will be saved to: {log_dir}")
        test(args, results_dir)
    else:
        print(f"Error: Model path '{args.model_path}' not found or invalid. Cannot proceed.")
        exit(1)



