"""
main.py 
-------
Main script to run training, evaluation, and interpretability experiments.
"""
import os
import cv2
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
import torchvision.transforms as transforms
import shap

# Import custom modules
from data_processing.csv_process import process_csv
from data_processing.dataset import MultiTaskImageDataset
from models.backbone import CustomBackbone
from models.multitask_model import MultiTaskModel
from interpretability.gradcam import GradCAM
from interpretability.lime import LIMEExplainer
from interpretability.shap import SHAPExplainer
from models.wrappers import SingleOutputWrapper
from utils.visualization import imshow_tensor

# Configuration and paths
CSV_PATH1 = '/fhome/pfeliu/tfg_feliu/TFG-Interpretability-Techniques-in-Social-Media-Images/data/39_20250401_0816.csv'
CSV_PATH2 = '/fhome/pfeliu/tfg_feliu/TFG-Interpretability-Techniques-in-Social-Media-Images/data/43_3.csv'
IMG_DIR = 'tfg_feliu/data/twitter'
OUTPUT_DIR = '/fhome/pfeliu/tfg_feliu/TFG-Interpretability-Techniques-in-Social-Media-Images/output'
AGREED_DF_PATH = '/fhome/pfeliu/tfg_feliu/TFG-Interpretability-Techniques-in-Social-Media-Images/data/X_labels_agreements_0204.csv'
MODEL_CHOICE = 'ResNet18'  # Options: ResNet18, EfficientNetB0, DenseNet121, ResNet50
SPANISH_DIR = '/fhome/pfeliu/tfg_feliu/data/spanish_dataset'

def prepare_data():
    df1 = process_csv(CSV_PATH1, IMG_DIR)
    df2 = process_csv(CSV_PATH2, IMG_DIR)
    agree_df = pd.concat([df1, df2], ignore_index=True)
    # Filter out specific unwanted landscape labels and remove duplicates
    agree_df = agree_df[~agree_df["landscape-type_visual"].isin([
        "forest_and_seminatural_areas,water_bodies", "artificial_surfaces,water_bodies", "artificial_surfaces,forest_and_seminatural_areas"])]
    agree_df = agree_df.drop_duplicates()
    print("El DataFrame de agreed tiene", len(agree_df), "filas.\n")
    agree_df.to_csv(AGREED_DF_PATH, index=False)
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

def build_dataset(transform):
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
    dataset = MultiTaskImageDataset(csv_file=AGREED_DF_PATH,
                                    img_dir=IMG_DIR,
                                    transform=transform,
                                    label_maps=label_maps)
    return dataset

def train_model(model, train_loader, device, num_epochs=10):
    # Losses per task with equal weights (modify if needed)
    weights_nature = torch.tensor([1.0, 1.0]).to(device)
    weights_materiality = torch.tensor([1.0, 1.0, 1.0]).to(device)
    weights_biological = torch.tensor([1.0, 1.0, 1.0]).to(device)
    weights_landscape = torch.tensor([1.0] * 8).to(device)
    
    criterion_nature = nn.CrossEntropyLoss(weight=weights_nature)
    criterion_materiality = nn.CrossEntropyLoss(weight=weights_materiality)
    criterion_biological = nn.CrossEntropyLoss(weight=weights_biological)
    criterion_landscape = nn.CrossEntropyLoss(weight=weights_landscape)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, image_names, labels in train_loader:
            images = images.to(device)
            labels_nature = labels['nature_visual'].to(device)
            labels_materiality = labels['nep_materiality_visual'].to(device)
            labels_biological = labels['nep_biological_visual'].to(device)
            labels_landscape = labels['landscape-type_visual'].to(device)
            
            optimizer.zero_grad()
            out_nature, out_materiality, out_biological, out_landscape = model(images)
            loss_nature = criterion_nature(out_nature, labels_nature)
            loss_materiality = criterion_materiality(out_materiality, labels_materiality)
            loss_biological = criterion_biological(out_biological, labels_biological)
            loss_landscape = criterion_landscape(out_landscape, labels_landscape)
            loss = loss_nature + loss_materiality + loss_biological + loss_landscape
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

def denormalize_image(tensor_image):
    """
    Converts a normalized tensor image back to a NumPy array in [0,1].
    """
    image = tensor_image.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    return np.clip(image, 0, 1)

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


def run_gradcam(model, test_dataset, device):
    gradcam_output_dir = os.path.join(OUTPUT_DIR, "grad_cam")
    os.makedirs(gradcam_output_dir, exist_ok=True)

    for img_idx in range(15):
        sample_image, sample_image_name, sample_labels = test_dataset[img_idx]
        input_tensor = sample_image.unsqueeze(0).to(device)
        # Assuming the target layer (for ResNet-based backbones) is layer4
        target_layer = model.backbone.backbone.layer4
        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate_cam(input_tensor)
        cam_resized = cv2.resize(cam, (224, 224))
        
        original_img = denormalize_image(sample_image)
        
        plt.imshow(original_img)
        plt.imshow(cam_resized, cmap='jet', alpha=0.5)
        plt.title("GradCAM - Nature Visual")
        plt.axis('off')

        gradcam_path = os.path.join(OUTPUT_DIR, "grad_cam", f"{sample_image_name}_gradcam.png")
        plt.savefig(gradcam_path, bbox_inches='tight')
        plt.close()
        print(f"GradCAM output saved to {gradcam_path}")

'''def run_shap(model, test_dataset, device):
    """
    Computes and saves SHAP explanation images for a subset of test samples.
    """
    shap_output_dir = os.path.join(OUTPUT_DIR, "shap")
    os.makedirs(shap_output_dir, exist_ok=True)

    # Build a background dataset using the first 10 images from the test dataset.
    background = []
    for i in range(10):
        sample_image, sample_image_name, _ = test_dataset[i]
        background.append(sample_image.unsqueeze(0))  # add a batch dimension
    background = torch.cat(background, dim=0).to(device)

    # Wrap the multi-task model so that it only outputs the desired branch (e.g., "nature_visual").
    from models.wrappers import SingleOutputWrapper
    single_model = SingleOutputWrapper(model, output_index=0)  # 0 corresponds to nature_visual

    # Pass the wrapped model (which now returns a single tensor) to the SHAP explainer.
    shap_explainer = SHAPExplainer(single_model, background, device)

    for img_idx in range(15):
        sample_image, sample_image_name, _ = test_dataset[img_idx]
        input_tensor = sample_image.unsqueeze(0).to(device)
        
        # Compute SHAP values for target task 0 (which corresponds to nature_visual).
        explanation = shap_explainer.explain(input_tensor, target_task=0)
        
        original_img = denormalize_image(sample_image)
        plt.imshow(original_img)
        plt.imshow(explanation, cmap='jet', alpha=0.5)
        plt.title("SHAP Explanation - Nature Visual")
        plt.axis('off')

        shap_path = os.path.join(shap_output_dir, f"{sample_image_name}_shap.png")
        plt.savefig(shap_path, bbox_inches='tight')
        plt.close()
        print(f"SHAP output saved to {shap_path}")'''


def run_shap(model, test_dataset, device):
    """
    Computes and saves SHAP explanation images for all tasks (4 tasks) for a subset of test samples.
    For each image, SHAP images for each task are saved.
    """
    shap_output_dir = os.path.join(OUTPUT_DIR, "shap_all")
    os.makedirs(shap_output_dir, exist_ok=True)

    # Build a background dataset using the first 10 images from the test dataset.
    background = []
    for i in range(10):
        sample_image, sample_image_name, _ = test_dataset[i]
        background.append(sample_image.unsqueeze(0))  # add a batch dimension
    background = torch.cat(background, dim=0).to(device)

    # Define a mapping from task index to task name.
    task_dict = {
        0: "nature_visual",
        1: "nep_materiality_visual",
        2: "nep_biological_visual",
        3: "landscape-type_visual"
    }

    # For each task, wrap the multi-task model so that it only returns the output for that task
    # and create a separate SHAPExplainer.
    from models.wrappers import SingleOutputWrapper
    shap_explainers = {}
    for task_idx, task_name in task_dict.items():
        wrapped_model = SingleOutputWrapper(model, output_index=task_idx)
        shap_explainers[task_idx] = SHAPExplainer(wrapped_model, background, device)

    # Process a subset of test samples.
    for img_idx in range(15):
        sample_image, sample_image_name, _ = test_dataset[img_idx]
        input_tensor = sample_image.unsqueeze(0).to(device)
        original_img = denormalize_image(sample_image)

        # Generate SHAP explanations for each task.
        for task_idx, task_name in task_dict.items():
            shap_explainer = shap_explainers[task_idx]
            # The target_task parameter is not needed here since our wrapped model has a single output,
            # but we keep it for compatibility.
            explanation = shap_explainer.explain(input_tensor, target_task=0)

            enhanced_exp = enhance_explanation(explanation, threshold=0.3, blur_kernel=(7, 7))
            
            plt.imshow(original_img)
            plt.imshow(enhanced_exp, cmap='jet', alpha=0.5)
            plt.title(f"SHAP Explanation - {task_name}")
            plt.axis('off')
            
            shap_path = os.path.join(shap_output_dir, f"{sample_image_name}_shap_{task_name}.png")
            plt.savefig(shap_path, bbox_inches='tight')
            plt.close()
            print(f"SHAP output saved to {shap_path}")

def run_lime(model, test_dataset, device):
    lime_output_dir = os.path.join(OUTPUT_DIR, "lime")
    os.makedirs(lime_output_dir, exist_ok=True)
    
    lime_explainer = LIMEExplainer(model, device)
    for idx in range(15):
        sample_image, sample_image_name, _ = test_dataset[idx]
        # Convert image tensor to a NumPy array in [0,1].
        image_np = denormalize_image(sample_image)
        
        save_path = os.path.join(lime_output_dir, f"{sample_image_name}_lime.png")
        lime_explainer.explain(image_np, target_task=0, save_path=save_path)
        print(f"LIME explanation saved to {save_path}")

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
    

import types
import torchvision.models.resnet as resnet

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


def main():
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Data Preparation
    agree_df = prepare_data()
    
    # Optional: Print unique label values and counts (for debugging)
    for col in ['nature_visual', 'nep_materiality_visual', 'nep_biological_visual', 'landscape-type_visual']:
        print(f"Etiquetas únicas en {col}:")
        print(agree_df[col].unique())
        print("------")
        print(agree_df[col].value_counts())
    
    transform = get_transforms()
    dataset = build_dataset(transform)
    print("\nTotal de imágenes:", len(dataset))
    
    # Split dataset into training and test sets.
    dataset_size = len(dataset)
    train_size = int(0.9 * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Step 2: Model Construction
    backbone = CustomBackbone(model_choice=MODEL_CHOICE)
    model = MultiTaskModel(backbone, backbone.feature_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Disable in-place ReLU operations
    disable_inplace_relu(model)
    # Patch the ResNet BasicBlocks to avoid in-place addition issues.
    patch_resnet_inplace(model)

    # Step 3: Training
    train_model(model, train_loader, device, num_epochs=5)

    # GradCAM Visualization on a few test samples.
    #run_gradcam(model, test_dataset, device)
    #print("\ngrad cam ejecutado")
    # SHAP explanation.
    #run_shap(model, test_dataset, device)
    #print("\nshap ejecutado")
    # LIME explanation.
    #run_lime(model, test_dataset, device)
    #print("\nlime ejecutado")

    # Step 5: Evaluation on Test Set.
    evaluate_model(model, test_loader, device)

if __name__ == '__main__':
    main()
