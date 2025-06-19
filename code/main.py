"""
main.py 
-------
Main script to run training, evaluation, and interpretability experiments.
"""
import os
import torch
import torchvision.transforms as transforms
import random

from torch.utils.data import DataLoader, random_split, Subset
from utils.metrics_utils import plot_training_curves
from utils.main_utils import (prepare_data, get_transforms, build_dataset, 
                              evaluate_model, disable_inplace_relu, patch_resnet_inplace, 
                              train_and_record, extended_evaluate)

from data_processing.unlabeled_dataset import UnlabeledImageDataset

from models.backbone import CustomBackbone
from models.multitask_model import MultiTaskModel

from interpretability.lime import run_lime
from interpretability.gradcam import apply_gradcam_all
from interpretability.xrai import apply_xrai_all
from interpretability.occlusion_sensitivity import apply_occlusion_all

# Configuration and paths
CSV_PATH1 = '/fhome/pfeliu/tfg_feliu/TFG-Interpretability-Techniques-in-Social-Media-Images/data_files/39_20250401_0816.csv'
CSV_PATH2 = '/fhome/pfeliu/tfg_feliu/TFG-Interpretability-Techniques-in-Social-Media-Images/data_files/43_3.csv'
CSV_PATH3 = '/fhome/pfeliu/tfg_feliu/TFG-Interpretability-Techniques-in-Social-Media-Images/data_files/51.csv'
IMG_DIR = '/fhome/pfeliu/tfg_feliu/data/training_dataset'
OUTPUT_DIR = '/fhome/pfeliu/tfg_feliu/TFG-Interpretability-Techniques-in-Social-Media-Images/output'
MODELS_PATH = '/fhome/pfeliu/tfg_feliu/TFG-Interpretability-Techniques-in-Social-Media-Images/trained_models'
AGREED_DF_PATH = '/fhome/pfeliu/tfg_feliu/TFG-Interpretability-Techniques-in-Social-Media-Images/data_files/X_labels_agreements_0205.csv'
SPANISH_DIR = '/fhome/pfeliu/tfg_feliu/data/spanish_dataset'
UNSEEN_DIR = '/fhome/pfeliu/tfg_feliu/data/interpretability_samples'

MODEL_CHOICE = 'ResNet18'  # Options: ResNet18, EfficientNetB0, DenseNet121, ResNet50
TRAIN = False
EPOCHS = 100
CHECKPOINT_PATH = os.path.join(MODELS_PATH, f'trained_{MODEL_CHOICE}_{EPOCHS}epochs.pth')

TARGET_LAYER = 'backbone.backbone.layer4'  #target layer for the different models
MODEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, MODEL_CHOICE)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

SEEN_SAMPLE_COUNT = 30
UNSEEN_SAMPLE_COUNT = 30


def main():
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Data Preparation
    agree_df = prepare_data(CSV_PATH1, CSV_PATH2, CSV_PATH3, IMG_DIR, AGREED_DF_PATH)
    
    task_names = ['nature_visual', 'nep_materiality_visual', 'nep_biological_visual', 'landscape-type_visual']
    for col in task_names:
        print(f"Etiquetas únicas en {col}:")
        print(agree_df[col].unique())
        print("------")
        print(agree_df[col].value_counts())
    
    transform = get_transforms()
    dataset = build_dataset(transform, IMG_DIR, AGREED_DF_PATH)
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

    if TRAIN:
        history = train_and_record(model, train_loader, test_loader, device, num_epochs=EPOCHS)
        
        #Save the trained model    
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"Model saved to {CHECKPOINT_PATH}")

        plot_training_curves(history, out_dir=os.path.join(MODEL_OUTPUT_DIR, 'figures'))
        metrics = extended_evaluate(model, test_loader, device, output_dir=os.path.join(MODEL_OUTPUT_DIR, 'evaluation'))
        print("\nEvaluation metrics:", metrics)

    else:
        if not os.path.isfile(CHECKPOINT_PATH):
            raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
        print(f"Loading weights from {CHECKPOINT_PATH}")
        state = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state)

    model.eval()


    random.seed(42)
    seen_idxs = random.sample(list(range(len(train_dataset))), SEEN_SAMPLE_COUNT)
    seen_subset = Subset(train_dataset, seen_idxs)
    seen_loader = DataLoader(seen_subset, batch_size=1, shuffle=False)

    interp_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    unseen_ds = UnlabeledImageDataset(UNSEEN_DIR, transform=interp_transform)
    if len(unseen_ds) > UNSEEN_SAMPLE_COUNT:
        unseen_ds = torch.utils.data.Subset(unseen_ds, list(range(UNSEEN_SAMPLE_COUNT)))

    unseen_loader = DataLoader(unseen_ds, batch_size=1, shuffle=False)


    # explainer functions
    explainers = {
        'GRADCAM': apply_gradcam_all,
        'XRAI': apply_xrai_all,
        'OCCLUSION_SENSITIVITY': apply_occlusion_all,
        'LIME': run_lime,

        #'SHAP': apply_shap_all,
        #'GUIDED_GRADCAM': apply_guided_gradcam_all,
    }

    for name_tech, function_call in explainers.items():  
        for split_name, loader in [('seen samples', seen_loader), ('unseen samples', unseen_loader)]: 
            out_dir = os.path.join(MODEL_OUTPUT_DIR, name_tech, split_name)
            os.makedirs(out_dir, exist_ok=True)
            print(f"Running {name_tech} on {split_name} samples → {out_dir}")

            if name_tech == 'LIME':
                ds = seen_subset if split_name=='seen samples' else unseen_ds
                function_call(model, ds, device, num_images=30, output_dir=out_dir)

            elif name_tech == 'OCCLUSION_SENSITIVITY':
                function_call(multi_model=model, device=device, data_loader=loader,
                    output_dir=out_dir, num_images=30, patch_size=32, stride=16, baseline=0.0, alpha=0.5)

            elif name_tech == 'XRAI':
                function_call(multi_model=model, device=device, data_loader=loader, output_dir=out_dir,
                    num_images=30, steps=50, alpha=0.5)

            elif name_tech == 'SHAP':
                function_call(multi_model=model, device=device, data_loader=loader,
                    output_dir=out_dir, num_images=30, background_size=100, alpha=0.5)

            else:
                function_call(multi_model=model, device=device, data_loader=loader,
                    output_dir=out_dir, num_images=30,
                    target_layer=TARGET_LAYER, alpha=0.5)


    # Step 5: Evaluation on Test Set.
    evaluate_model(model, test_loader, device)

if __name__ == '__main__':
    main()
