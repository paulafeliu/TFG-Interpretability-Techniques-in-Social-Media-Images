# metrics_utils.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    f1_score, balanced_accuracy_score, matthews_corrcoef,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import torch

def plot_training_curves(history: dict, out_dir: str):
    """
    history should be a dict with keys:
      'train_loss', 'val_loss', 'train_acc_<task>', 'val_acc_<task>'...
    """
    os.makedirs(out_dir, exist_ok=True)
    # Loss
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'],   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig(os.path.join(out_dir, 'loss_curve.png'), dpi=150)
    plt.close()

    # Accuracy for each task
    for task in [k.replace('train_acc_','') for k in history if k.startswith('train_acc_')]:
        plt.figure()
        plt.plot(history[f'train_acc_{task}'], label='Train')
        plt.plot(history[f'val_acc_{task}'],   label='Val')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy')
        plt.title(task)
        plt.legend()
        plt.savefig(os.path.join(out_dir, f'acc_curve_{task}.png'), dpi=150)
        plt.close()

def compute_extended_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_name: str):
    """Compute macro/micro F1, balanced accuracy, MCC for a single task."""
    metrics = {
        'task': task_name,
        'f1_macro':     f1_score(y_true, y_pred, average='macro'),
        'f1_micro':     f1_score(y_true, y_pred, average='micro'),
        'bal_accuracy': balanced_accuracy_score(y_true, y_pred),
        'mcc':          matthews_corrcoef(y_true, y_pred)
    }
    return metrics

def plot_confusion_heatmap(y_true: np.ndarray, y_pred: np.ndarray, labels: list, out_path: str):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.title('Confusion matrix (normalized)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

'''def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_classes: int,
    task_name: str,
    out_dir: str
):
    """For binary: n_classes=2; for multi-class we do one-vs-rest."""
    os.makedirs(out_dir, exist_ok=True)
    # Binarize labels
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:,i], y_score[:,i])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC={roc_auc:.2f}')
        plt.plot([0,1],[0,1],'--')
        plt.xlabel('FPR'); plt.ylabel('TPR')
        plt.title(f'ROC curve: {task_name} (class {i})')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(out_dir, f'roc_{task_name}_cls{i}.png'), dpi=150)
        plt.close()

        precision, recall, _ = precision_recall_curve(y_bin[:,i], y_score[:,i])
        pr_auc = auc(recall, precision)
        plt.figure()
        plt.plot(recall, precision, label=f'AP={pr_auc:.2f}')
        plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.title(f'PR curve: {task_name} (class {i})')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(out_dir, f'pr_{task_name}_cls{i}.png'), dpi=150)
        plt.close()'''

def tsne_feature_visualization(
    feature_extractor: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    task_labels: dict,
    out_path: str,
    n_samples: int = 1000
):
    """
    Extract features from backbone, run t-SNE and plot colored by one of the tasks.
    task_labels: dict mapping image-index → label for coloring.
    """
    features = []
    labels   = []
    with torch.no_grad():
        for i, (imgs, img_names, labs) in enumerate(data_loader):
            if i >= n_samples: break
            feat = feature_extractor(imgs.to(device)).cpu().numpy()
            features.append(feat)
            labels.append(labs[list(task_labels.keys())[0]].numpy())  # first task
    X = np.vstack(features)
    y = np.concatenate(labels)

    tsne = TSNE(n_components=2, random_state=42)
    X2 = tsne.fit_transform(X)

    plt.figure(figsize=(8,6))
    palette = sns.color_palette("hsv", len(np.unique(y)))
    sns.scatterplot(x=X2[:,0], y=X2[:,1], hue=y, palette=palette, legend='full', s=10)
    plt.title('t-SNE of learned features')
    plt.savefig(out_path, dpi=150)
    plt.close()
