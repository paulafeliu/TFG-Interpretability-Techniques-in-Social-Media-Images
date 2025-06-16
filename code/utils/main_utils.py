"""
utils.py 
-------
Funciones auxiliares para entrenamiento, evaluación e interpretabilidad.
"""

import os
import types
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
import torchvision.transforms as transforms
import torchvision.models.resnet as resnet
from sklearn.metrics import confusion_matrix, classification_report

import torchvision.transforms as transforms
import types
import torchvision.models.resnet as resnet

from utils.metrics_utils import (compute_extended_metrics,
    plot_confusion_heatmap)

from data_processing.csv_process import process_csv
from data_processing.csv_process import process_csv3
from data_processing.dataset import MultiTaskImageDataset


def prepare_data(csv1, csv2, csv3, img_dir, agreed_df_path):
    df1 = process_csv(csv1, img_dir)
    df2 = process_csv(csv2, img_dir)
    df3 = process_csv3(csv3, img_dir)
    agree_df = pd.concat([df1, df2, df3], ignore_index=True)
    # Filter out specific unwanted landscape labels and remove duplicates
    agree_df = agree_df[~agree_df["landscape-type_visual"].isin([
        "forest_and_seminatural_areas,water_bodies", "artificial_surfaces,water_bodies", "artificial_surfaces,forest_and_seminatural_areas"])]
    agree_df = agree_df.drop_duplicates()
    print("El DataFrame de agreed tiene", len(agree_df), "filas.\n")
    agree_df.to_csv(agreed_df_path, index=False)
    return agree_df

def get_transforms():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform

def build_dataset(transform, img_dir, agreed_df_path):
    # Define label maps for tasks
    label_maps = {
        "nature_visual": {"Yes": 1, "No": 0},
        "nep_materiality_visual": {"material": 0, "immaterial": 1, 'nan': 2},
        "nep_biological_visual": {"biotic": 0, "abiotic": 1, 'nan': 2},
        "landscape-type_visual": {
            "artificial_surfaces": 0,
            "forest_and_seminatural_areas": 1,
            "wetlands": 2,
            "water_bodies": 3,
            "agricultural_areas": 4,
            "other": 5,
            "none": 6, 
            'nan': 7
        }
    }
    dataset = MultiTaskImageDataset(csv_file=agreed_df_path,
                                    img_dir=img_dir,
                                    transform=transform,
                                    label_maps=label_maps)
    return dataset

def enhance_explanation(explanation, threshold=0.3, blur_kernel=(7, 7)):
    """
    Enhance the explanation heatmap by thresholding and smoothing.
    
    Parameters:
        explanation: A normalized heatmap with values in [0, 1].
        threshold: A float value; heatmap values below this are set to 0.
        blur_kernel: Kernel size for Gaussian blur.
        
    Returns:
        An enhanced heatmap as a numpy array.
    """
    # Threshold: remove low activation (set these to zero)
    enhanced = explanation.copy()
    enhanced[enhanced < threshold] = 0.0
    
    # Optionally, amplify the high activation regions (contrast stretching)
    # For example, multiply by a factor to boost the values.
    enhanced = np.clip(enhanced * 1.5, 0, 1)
    
    # Gaussian Blur: smooth the output to create contiguous regions.
    enhanced = cv2.GaussianBlur(enhanced, blur_kernel, 0)
    
    return enhanced

def evaluate_model(model, test_loader, device):
    weights_nature = torch.tensor([1.0, 1.0]).to(device)
    weights_materiality = torch.tensor([1.0, 1.0, 1.0]).to(device)
    weights_biological = torch.tensor([1.0, 1.0, 1.0]).to(device)
    weights_landscape = torch.tensor([1.0] * 8).to(device)
    
    criterion_nature = nn.CrossEntropyLoss(weight=weights_nature)
    criterion_materiality = nn.CrossEntropyLoss(weight=weights_materiality)
    criterion_biological = nn.CrossEntropyLoss(weight=weights_biological)
    criterion_landscape = nn.CrossEntropyLoss(weight=weights_landscape)
    
    model.eval()
    test_loss = 0.0
    total = 0

    all_preds = {k: [] for k in ['nature_visual', 'nep_materiality_visual', 'nep_biological_visual', 'landscape-type_visual']}
    all_labels = {k: [] for k in ['nature_visual', 'nep_materiality_visual', 'nep_biological_visual', 'landscape-type_visual']}
    
    with torch.no_grad():
        for images, image_names, labels in test_loader:
            images = images.to(device)
            labels_nature = labels['nature_visual'].to(device)
            labels_materiality = labels['nep_materiality_visual'].to(device)
            labels_biological = labels['nep_biological_visual'].to(device)
            labels_landscape = labels['landscape-type_visual'].to(device)
            
            out_nature, out_materiality, out_biological, out_landscape = model(images)
            loss = (criterion_nature(out_nature, labels_nature) + 
                    criterion_materiality(out_materiality, labels_materiality) +
                    criterion_biological(out_biological, labels_biological) +
                    criterion_landscape(out_landscape, labels_landscape))
            test_loss += loss.item()
            
            preds_nature = out_nature.argmax(dim=1)
            preds_materiality = out_materiality.argmax(dim=1)
            preds_biological = out_biological.argmax(dim=1)
            preds_landscape = out_landscape.argmax(dim=1)
            
            all_preds['nature_visual'].extend(preds_nature.cpu().numpy())
            all_preds['nep_materiality_visual'].extend(preds_materiality.cpu().numpy())
            all_preds['nep_biological_visual'].extend(preds_biological.cpu().numpy())
            all_preds['landscape-type_visual'].extend(preds_landscape.cpu().numpy())
            
            all_labels['nature_visual'].extend(labels_nature.cpu().numpy())
            all_labels['nep_materiality_visual'].extend(labels_materiality.cpu().numpy())
            all_labels['nep_biological_visual'].extend(labels_biological.cpu().numpy())
            all_labels['landscape-type_visual'].extend(labels_landscape.cpu().numpy())
            
            total += images.size(0)
    
    avg_test_loss = test_loss / len(test_loader)
    print("\n----------------- EVALUATION -----------------")
    print(f"\nTest Loss: {avg_test_loss:.4f}")
    
    # Print accuracy per task
    for task in all_preds.keys():
        acc = np.mean(np.array(all_preds[task]) == np.array(all_labels[task]))
        print(f"Accuracy for {task}: {acc*100:.2f}%")
        print(f"\nConfusion Matrix - {task}:")
        print(confusion_matrix(all_labels[task], all_preds[task]))
        print(f"\nClassification Report - {task}:")
        print(classification_report(all_labels[task], all_preds[task]))

def disable_inplace_relu(model):
    """
    Recorre el modelo y desactiva la operación in-place en todas las capas ReLU.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False
    
def patch_resnet_inplace(model):
    """
    Patches all BasicBlock modules in the model so that in-place additions
    are replaced by out-of-place operations. This circumvents SHAP's issue
    with modifying views in-place during gradient computation.
    """
    for module in model.modules():
        if isinstance(module, resnet.BasicBlock):
            # Avoid repatching if already patched.
            if not hasattr(module, 'patched_for_shap'):
                def new_forward(self, x):
                    identity = x
                    out = self.conv1(x)
                    out = self.bn1(out)
                    out = self.relu(out)
                    out = self.conv2(out)
                    out = self.bn2(out)
                    if self.downsample is not None:
                        identity = self.downsample(x)
                    # Replace in-place addition with an out-of-place addition.
                    out = out + identity  
                    out = self.relu(out)
                    return out
                module.forward = types.MethodType(new_forward, module)
                module.patched_for_shap = True

def train_and_record(model, train_loader, val_loader, device, num_epochs):
    history = {
        'train_loss': [], 'val_loss': [],
        # will dynamically add train_acc_<task>, val_acc_<task>
    }
    # initialize per-epoch lists
    for task in ['nature_visual', 'nep_materiality_visual', 'nep_biological_visual', 'landscape-type_visual']:
        history[f'train_acc_{task}'] = []
        history[f'val_acc_{task}']   = []

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterions = {
      'nature_visual': nn.CrossEntropyLoss(),
      'nep_materiality_visual': nn.CrossEntropyLoss(),
      'nep_biological_visual': nn.CrossEntropyLoss(),
      'landscape-type_visual': nn.CrossEntropyLoss()
    }

    for epoch in range(num_epochs):
        # TRAIN
        model.train()
        running_loss = 0
        correct = {t: 0 for t in criterions}
        total   = 0

        for images, _, labels in train_loader:
            images = images.to(device)
            outs = model(images)
            loss = sum(criterions[t](outs[i], labels[list(criterions.keys())[i]].to(device))
                       for i,t in enumerate(criterions))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running_loss += loss.item()
            preds = [out.argmax(1).cpu().numpy() for out in outs]
            for i,t in enumerate(criterions):
                correct[t] += (preds[i] == labels[t].numpy()).sum()
            total += images.size(0)

        history['train_loss'].append(running_loss/len(train_loader))
        for t in criterions:
            history[f'train_acc_{t}'].append(correct[t]/total)

        # VALIDATION (same pattern)
        model.eval()
        val_loss = 0
        correct = {t: 0 for t in criterions}
        total   = 0
        with torch.no_grad():
            for images, _, labels in val_loader:
                images = images.to(device)
                outs = model(images)
                loss = sum(criterions[t](outs[i], labels[list(criterions.keys())[i]].to(device))
                           for i,t in enumerate(criterions))
                val_loss += loss.item()
                preds = [out.argmax(1).cpu().numpy() for out in outs]
                for i,t in enumerate(criterions):
                    correct[t] += (preds[i] == labels[t].numpy()).sum()
                total += images.size(0)

        history['val_loss'].append(val_loss/len(val_loader))
        for t in criterions:
            history[f'val_acc_{t}'].append(correct[t]/total)

        print(f"Epoch {epoch+1}: "
              f"Train Loss {history['train_loss'][-1]:.4f}, "
              f"Val Loss {history['val_loss'][-1]:.4f}")

    return history

def extended_evaluate(model, test_loader, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # If the dataset is wrapped in a Subset, unwrap it
    raw_ds = test_loader.dataset
    if isinstance(raw_ds, Subset):
        base_ds = raw_ds.dataset
    else:
        base_ds = raw_ds

    tasks = [
        'nature_visual',
        'nep_materiality_visual',
        'nep_biological_visual',
        'landscape-type_visual'
    ]
    all_true, all_pred, all_scores = {t: [] for t in tasks}, {t: [] for t in tasks}, {t: [] for t in tasks}

    model.eval()
    with torch.no_grad():
        for images, _, labels in test_loader:
            images = images.to(device)
            outs = model(images)  # tuple of logits per task

            for i, t in enumerate(tasks):
                logits = outs[i].cpu().numpy()
                preds = logits.argmax(axis=1)
                all_true[t].extend(labels[t].numpy())
                all_pred[t].extend(preds)
                all_scores[t].extend(logits)

    metrics_list = []
    for t in tasks:
        # 1 compute extended metrics
        m = compute_extended_metrics(
            np.array(all_true[t]),
            np.array(all_pred[t]),
            task_name=t
        )
        metrics_list.append(m)

        # 2 confusion‐matrix heatmap
        label_names = list(base_ds.label_maps[t].keys())
        plot_confusion_heatmap(
            np.array(all_true[t]),
            np.array(all_pred[t]),
            labels=label_names,
            out_path=os.path.join(output_dir, f'cm_{t}.png')
        )

    return metrics_list
    

