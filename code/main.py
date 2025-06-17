"""
main.py 
-------
Main script to run training, evaluation, and interpretability experiments.
"""
import json
import os
import types
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models.resnet as resnet
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm

try:
    from code.settings import CSV_PATH1, IMG_DIR, CSV_PATH2, AGREED_DF_PATH, OUTPUT_DIR, MODEL_CHOICE, CHECKPOINT_PATH, \
        NUM_EPOCHS
except ImportError:
    print("No code.settings module found. Copy, rename and adapt the template_settings.py file")
    exit(1)
# Import custom modules
from data_processing.csv_process import process_csv
from data_processing.dataset import MultiTaskImageDataset, InferenceImageDataset
# from interpretability.gradcam_all import run_gradcam_all
from interpretability.lime import run_lime
from interpretability.shap import SHAPExplainer
from models.backbone import CustomBackbone
from models.multitask_model import MultiTaskModel
# from gradcam_backprop import GradCAM, GuidedBackprop
from utils.denormalize_image import denormalize_image
from torch._C import device as TorchDevice
import torch.nn.functional as F


def prepare_data():
    if not IMG_DIR:
        raise FileNotFoundError('No image directory provided.')
    df1 = process_csv(CSV_PATH1, IMG_DIR)
    df2 = process_csv(CSV_PATH2, IMG_DIR)
    agree_df = pd.concat([df1, df2], ignore_index=True)
    # Filter out specific unwanted landscape labels and remove duplicates
    agree_df = agree_df[~agree_df["landscape-type_visual"].isin([
        "forest_and_seminatural_areas,water_bodies", "artificial_surfaces,water_bodies",
        "artificial_surfaces,forest_and_seminatural_areas"])]
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

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")
    # Saving the model
    torch.save(model.state_dict(), CHECKPOINT_PATH)


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

    all_preds = {k: [] for k in
                 ['nature_visual', 'nep_materiality_visual', 'nep_biological_visual', 'landscape-type_visual']}
    all_labels = {k: [] for k in
                  ['nature_visual', 'nep_materiality_visual', 'nep_biological_visual', 'landscape-type_visual']}

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
        print(f"Accuracy for {task}: {acc * 100:.2f}%")
        print(f"\nConfusion Matrix - {task}:")
        print(confusion_matrix(all_labels[task], all_preds[task]))
        print(f"\nClassification Report - {task}:")
        print(classification_report(all_labels[task], all_preds[task]))


def inference(model: MultiTaskModel, device: str, dataset: InferenceImageDataset):
    model.eval()
    feature_names = ["nature", "materiality", "biological", "landscape"]
    all_results = {k: [] for k in feature_names}

    dataset_loader = DataLoader(dataset)

    with torch.no_grad():
        for images in tqdm(dataset_loader):
            images = images.to(device)

            outputs = model(images)

            for feature_name, output in zip(feature_names, outputs):
                selected_class = output.argmax(dim=1).cpu().item()
                probability = F.softmax(output, dim=1).max(dim=1)[0].cpu().item()
                all_results[feature_name].append((selected_class, probability))

    return all_results


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


def prepare_dataset() -> tuple[Subset, Subset, DataLoader, DataLoader]:
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
    return train_dataset, test_dataset, train_loader, test_loader


def prepare_model(device: TorchDevice, checkpoint_path: Optional[str] = None) -> MultiTaskModel:
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    backbone = CustomBackbone(model_choice=MODEL_CHOICE)
    model = MultiTaskModel(backbone, backbone.feature_dim)

    model = model.to(device)

    # Disable in-place ReLU operations
    disable_inplace_relu(model)
    # Patch the ResNet BasicBlocks to avoid in-place addition issues.
    patch_resnet_inplace(model)

    if checkpoint_path:
        print("preload checkpoint")
        model_set_checkpoint(model, device, checkpoint_path)
    return model


def model_set_checkpoint(model: MultiTaskModel,
                         device: TorchDevice,
                         checkpoint_path: str,
                         set_eval_mode: bool = True) -> None:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"Loading weights from {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    if set_eval_mode:
        model.eval()


def main(checkpoint_path: str,
         train: bool = False,
         preload_checkpoint: bool = False,
         run_interpret_gradcam: bool = False,
         run_interpret_shap: bool = False,
         run_interpret_lime: bool = False,
         run_eval: bool = False, ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 2: Model Construction
    model = prepare_model(device, checkpoint_path if preload_checkpoint else None)

    # Step 3: Training
    if train:
        # Step 1: Data Preparation
        train_dataset, test_dataset, train_loader, test_loader = prepare_dataset()

        try:
            train_model(model, train_loader, device, num_epochs=NUM_EPOCHS)
        except KeyboardInterrupt:
            save_prompt = input("Training interrupted. Save model? [any key / n]")
            if save_prompt != "n":
                print(f"Saving model to {CHECKPOINT_PATH}")
                torch.save(model.state_dict(), CHECKPOINT_PATH)
            print("Saliendo")
            exit(0)

    model_set_checkpoint(model, device, checkpoint_path)

    # ... check run_interpret_gradcam
    # GradCAM Visualization on a few test samples.
    # run_gradcam(model, test_dataset, device)

    # parameter?
    # Grad-CAM para todas las tasks:
    '''run_gradcam_all(
        multi_model=model,
        test_dataset=test_dataset,
        device=device,
        output_dir=OUTPUT_DIR,
        num_images=15,  # o la cantidad de ejemplos que prefieras
        alpha=0.5
    )
    print("\nGrad-CAM para todas las tareas ejecutado.")'''
    # print("\ngrad cam ejecutado")

    # ... check run_interpret_shap
    # SHAP explanation.
    # run_shap(model, test_dataset, device)
    # print("\nshap ejecutado")

    # LIME explanation.
    if run_interpret_lime:
        lime_output_dir = os.path.join(OUTPUT_DIR, "lime_all")
        os.makedirs(lime_output_dir, exist_ok=True)
        run_lime(model, test_dataset, device, num_images=10, output_dir=lime_output_dir)
        # print("\nlime ejecutado")

    '''
    SHAP EXPLANATION
    shap_output_dir = os.path.join(OUTPUT_DIR, "shap2")
    os.makedirs(shap_output_dir, exist_ok=True)
    shap_analysis(model, test_dataset, device, num_images=10, output_dir=shap_output_dir)'''

    # gradcam_backprop_dir = os.path.join(OUTPUT_DIR, "gradcam_backpropagation")

    '''
    GUIDED GRADCAM
    
    guided_dir = os.path.join(OUTPUT_DIR, "guided_gradcam")
    #print(f"\nRunning Grad‐CAM on {args.task}, saving to {gradcam_dir}")
    apply_guided_gradcam_all(
        multi_model=model,
        device=device,
        data_loader=test_loader,
        output_dir=guided_dir,
        num_images=30,                         # or however many per task
        target_layer="model.backbone.backbone.layer4",
        alpha=0.5
    )'''

    # Step 5: Evaluation on Test Set.
    if run_eval:
        evaluate_model(model, test_loader, device)


if __name__ == '__main__':
    # main(CHECKPOINT_PATH,
    #      train=False,
    #      preload_checkpoint=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = prepare_model(device, CHECKPOINT_PATH)

    ds = InferenceImageDataset(Path("/home/rsoleyma/projects/big5/labelstudio-tools/data/temp/snippets"),
                               get_transforms())
    predictions = inference(model, device, ds)
    json.dump(predictions, Path("data/predictions.json").open("w"))
    print(predictions)
