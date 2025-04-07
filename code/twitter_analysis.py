#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
import copy
from sklearn.metrics import confusion_matrix, classification_report
import cv2

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

def process_csv(file_path, img_dir, valid_extensions=['.jpg', '.png', '.jpeg']):
    df = pd.read_csv(file_path)
    new_rows = []
    for idx in range(4):
        material_col = f'nep_materiality_visual_{idx}'
        biological_col = f'nep_biological_visual_{idx}'
        landscape_col = f'landscape-type_visual_{idx}'
        
        temp_df = df[['platform_id', 'nature_visual', material_col, biological_col, landscape_col]].copy()
        temp_df['index'] = idx
        temp_df = temp_df.rename(columns={
            material_col: 'nep_materiality_visual',
            biological_col: 'nep_biological_visual',
            landscape_col: 'landscape-type_visual'
        })
        temp_df['image_name'] = temp_df['platform_id'].astype(str) + '_' + temp_df['index'].astype(str)
        
        # Filtrar solo las filas cuya imagen existe en el directorio
        def image_exists(row):
            base_img_name = row['image_name']
            for ext in valid_extensions:
                img_path = os.path.join(img_dir, base_img_name + ext).replace('\\', '/')
                if os.path.exists(img_path):
                    return True
            return False
        
        temp_df = temp_df[temp_df.apply(image_exists, axis=1)]

        new_rows.append(temp_df)
        
    
    new_df = pd.concat(new_rows, ignore_index=True)
    #new_df = new_df.dropna(subset=['nep_materiality_visual', 'nep_biological_visual'])
    new_df = new_df.sort_values(by=['platform_id', 'index']).reset_index(drop=True)
    
    def process_nature(val):
        if pd.isna(val):
            return np.nan
        items = [item.strip() for item in str(val).split(';')]
        for item in items:
            if item.lower() == "yes":
                return "Yes"
        return "No"
    
    def process_other(val):
        if pd.isna(val):
            return np.nan
        items = [item.strip() for item in str(val).split(';')]
        if len(items) == 1:
            return items[0]
        if len(set(items)) == 1:
            return items[0]
        return "; ".join(items)
    
    new_df['nature_visual'] = new_df['nature_visual'].apply(process_nature)
    new_df['nep_materiality_visual'] = new_df['nep_materiality_visual'].apply(process_other)
    new_df['nep_biological_visual'] = new_df['nep_biological_visual'].apply(process_other)
    new_df['landscape-type_visual'] = new_df['landscape-type_visual'].apply(process_other)

    #-------------------------------
    mask = (new_df['nature_visual'] == "No") | (
        (new_df['nature_visual'] == "Yes") &
        new_df['nep_materiality_visual'].notna() &
        new_df['nep_biological_visual'].notna() 
        #new_df['landscape-type_visual'].notna()
    )
    #new_df = new_df[mask]
    masked_df = new_df[mask]
    #-------------------------------
    
    new_order = ['image_name', 'nature_visual', 'nep_materiality_visual', 'nep_biological_visual', 
                 'landscape-type_visual', 'platform_id', 'index']
    masked_df = masked_df[new_order]
    
    def tiene_contradicciones(valor):
        if pd.isna(valor):
            return False
        items = [item.strip() for item in str(valor).split(';')]
        return len(set(items)) > 1

    cols = ['nep_materiality_visual', 'nep_biological_visual', 'landscape-type_visual'] #'nature_visual'
    df_bool = masked_df[cols].applymap(tiene_contradicciones)
    agreed_df = masked_df[~df_bool.any(axis=1)]
    print("El DataFrame tiene", len(agreed_df), "filas.")
    
    return agreed_df


class MultiTaskImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, label_maps=None):
        self.img_dir = img_dir
        self.transform = transform
        self.label_maps = label_maps

        # Read the CSV file containing image references.
        data = pd.read_csv(csv_file).fillna("nan")
        extensions = ['.jpg', '.png', '.jpeg']
        valid_rows = []

        # Iterate over each row in the CSV file.
        for idx, row in data.iterrows():
            # Construct the base image name (e.g., "platform_id_index")
            base_img_name = row['image_name']
            # Check if at least one image file with the expected extensions exists.
            if any(os.path.exists(os.path.join(self.img_dir, base_img_name + ext).replace('\\', '/')) 
                   for ext in extensions):
                valid_rows.append(row)
            # If the image file doesn't exist, the row is not added.

        # Only keep the rows with valid image files.
        self.data = pd.DataFrame(valid_rows).reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        base_img_name = row['image_name']
        extensions = ['.jpg', '.png', '.jpeg']
        img_path = None

        # Find the correct image file.
        for ext in extensions:
            candidate = os.path.join(self.img_dir, base_img_name + ext).replace('\\', '/')
            if os.path.exists(candidate):
                img_path = candidate
                break
        
        # Since we filtered earlier, img_path should always be valid.
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        labels = {}
        for task in ['nature_visual', 'nep_materiality_visual', 'nep_biological_visual', 'landscape-type_visual']:
            label_value = row[task]
            if self.label_maps and task in self.label_maps:
                label_value = self.label_maps[task].get(label_value, -1)
            labels[task] = label_value

        return image, row['image_name'], labels



def imshow_tensor(image_tensor, title=None, save_path=None):
    image = image_tensor.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    plt.imshow(image)
    if title:
        plt.title(title, fontsize=8)
    plt.axis('off')
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


# Pretrained Models Backbone ------------------------------------------

'''class ResNet50Backbone(nn.Module):
    def __init__(self, backbone):
        super(ResNet50Backbone, self).__init__()
        self.backbone = backbone
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return x'''

class CustomBackbone(nn.Module):
    def __init__(self, model_choice='DenseNet121'):
        super(CustomBackbone, self).__init__()
        self.model_choice = model_choice
        if model_choice == 'DenseNet121':
            model_base = models.densenet121(pretrained=True)
            # Remove classifier by replacing with identity
            model_base.classifier = nn.Identity()
            self.feature_dim = 1024
        elif model_choice == 'ResNet18':
            model_base = models.resnet18(pretrained=True)
            model_base.fc = nn.Identity()
            self.feature_dim = 512
        elif model_choice == 'EfficientNetB0':
            model_base = models.efficientnet_b0(pretrained=True)
            model_base.classifier = nn.Identity()
            self.feature_dim = 1280
        else:
            # Default to ResNet50
            model_base = models.resnet50(pretrained=True)
            model_base.fc = nn.Identity()
            self.feature_dim = 2048
        self.backbone = model_base

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return x

# --------------------------------------------------------------------------

class MultiTaskModel(nn.Module):
    def __init__(self, backbone, feature_dim=2048):
        super(MultiTaskModel, self).__init__()
        self.backbone = backbone
        self.fc_nature = nn.Linear(feature_dim, 3)
        self.fc_materiality = nn.Linear(feature_dim, 3)
        self.fc_biological = nn.Linear(feature_dim, 3)
        self.fc_landscape = nn.Linear(feature_dim, 8)
        
    def forward(self, x):
        features = self.backbone(x)
        out_nature = self.fc_nature(features)
        out_materiality = self.fc_materiality(features)
        out_biological = self.fc_biological(features)
        out_landscape = self.fc_landscape(features)
        return out_nature, out_materiality, out_biological, out_landscape


# ------------------------------ Grad-CAM Module ------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        """
        model: the full model (MultiTaskModel)
        target_layer: the convolutional layer to hook (e.g., model.backbone.backbone.layer4 for ResNet50)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks on the target layer
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        # grad_output is a tuple with gradients with respect to the output
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class=None):
        """
        input_tensor: tensor of shape (1, C, H, W)
        target_class: integer index of the class in the selected head (default: predicted class)
        Returns: cam of shape (H, W) (normalized)
        """
        # Forward pass: note that our model outputs a tuple for multiple tasks.
        outputs = self.model(input_tensor)
        # Select the output for one task (here, the "nature_visual" head, which is outputs[0])
        score = outputs[0]
        if target_class is None:
            target_class = score.argmax(dim=1).item()
        # Create one-hot vector for the target class for the first sample.
        one_hot = torch.zeros_like(score)
        one_hot[0, target_class] = 1
        self.model.zero_grad()
        score.backward(gradient=one_hot, retain_graph=True)
        # Global average pooling on gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        # Normalize the CAM
        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        # Remove batch and channel dimensions and return numpy array
        return cam.cpu().squeeze().numpy()
# ------------------------------------------------------------------------------







def main():
    #print("starting")
    
    csv_path1 = '/fhome/pfeliu/tfg_feliu/data/39_20250401_0816.csv'
    csv_path2 = '/fhome/pfeliu/tfg_feliu/data/43_2.csv'
    
    img_dir = '/fhome/pfeliu/tfg_feliu/data/twitter'
    output_dir = "/fhome/pfeliu/tfg_feliu/data/output" 

    df1 = process_csv(csv_path1, img_dir)
    df2 = process_csv(csv_path2, img_dir)

    #print("csv processed")

    agree_df = pd.concat([df1, df2], ignore_index=True)

    agree_df = agree_df[~agree_df["landscape-type_visual"].isin(["forest_and_seminatural_areas,water_bodies", "artificial_surfaces,water_bodies"])]
    agree_df = agree_df.drop_duplicates()
    print("\El DataFrame de agreed tiene", len(agree_df), "filas.\n")

    #print("agreed_df created") 

    agreed_df_path = '/fhome/pfeliu/tfg_feliu/data/X_labels_agreements_0204.csv'

    agree_df.to_csv(agreed_df_path, index=False)
    
    #print("agreed_df saved")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
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

    for col in ['nature_visual', 'nep_materiality_visual', 'nep_biological_visual', 'landscape-type_visual']:
        print(f"Etiquetas únicas en {col}:")
        print(agree_df[col].unique())
        print("------")
        print(agree_df[col].value_counts())
    
    dataset_csv = agreed_df_path
    
    dataset = MultiTaskImageDataset(csv_file=dataset_csv, img_dir=img_dir, transform=transform, label_maps=label_maps)
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print("\nTotal de imágenes:", len(dataset))
    print("Total de batches:", len(dataloader), "\n")
    
     
    os.makedirs(output_dir, exist_ok=True)

    output_imgs_dir = os.path.join(output_dir, "X_imgs")
    os.makedirs(output_imgs_dir, exist_ok=True)

    for idx in range(10):
        sample_image, sample_image_name, sample_labels = dataset[idx]
        title = (f"Name: {sample_image_name}\n"
                f"Labels: {sample_labels} ")
                
        save_path = os.path.join(output_imgs_dir, f"{sample_image_name}.png")
        imshow_tensor(sample_image, title=title, save_path=save_path)
    

    # Change pretrained model --------------------------------------------------------------

    model_choice = 'ResNet18'  # ResNet18, EfficientNetB0, DenseNet121, ResNet50
    backbone_model = CustomBackbone(model_choice=model_choice)
    feature_dim = backbone_model.feature_dim
    model = MultiTaskModel(backbone_model, feature_dim=feature_dim)
    
    # -------------------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    dataset_size = len(dataset)
    train_size = int(0.9 * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    

    # LOSS BALANCING ------------------------------------------------------------------------

    ##criterion = nn.CrossEntropyLoss()


    weights_nature = torch.tensor([1.0, 1.0, 1.0]).to(device)
    weights_materiality = torch.tensor([1.0, 1.0, 1.0]).to(device)
    weights_biological = torch.tensor([1.0, 1.0, 1.0]).to(device)
    weights_landscape = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(device)
    
    criterion_nature = nn.CrossEntropyLoss(weight=weights_nature)
    criterion_materiality = nn.CrossEntropyLoss(weight=weights_materiality)
    criterion_biological = nn.CrossEntropyLoss(weight=weights_biological)
    criterion_landscape = nn.CrossEntropyLoss(weight=weights_landscape)

    # ---------------------------------------------------------------------------------------
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
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
    

    # 3. CONFUSION MATRICES & ADDITIONAL METRICS ---------------------------------------------

    '''model.eval()
    test_loss = 0.0
    correct_nature = 0
    correct_materiality = 0
    correct_biological = 0
    correct_landscape = 0
    total = 0
    
    with torch.no_grad():
        for images, image_names, labels in test_loader:
            images = images.to(device)
            labels_nature = labels['nature_visual'].to(device)
            labels_materiality = labels['nep_materiality_visual'].to(device)
            labels_biological = labels['nep_biological_visual'].to(device)
            labels_landscape = labels['landscape-type_visual'].to(device)
            
            out_nature, out_materiality, out_biological, out_landscape = model(images)
            loss_nature = criterion(out_nature, labels_nature)
            loss_materiality = criterion(out_materiality, labels_materiality)
            loss_biological = criterion(out_biological, labels_biological)
            loss_landscape = criterion(out_landscape, labels_landscape)
            loss = loss_nature + loss_materiality + loss_biological + loss_landscape
            test_loss += loss.item()
            
            preds_nature = out_nature.argmax(dim=1)
            preds_materiality = out_materiality.argmax(dim=1)
            preds_biological = out_biological.argmax(dim=1)
            preds_landscape = out_landscape.argmax(dim=1)
            
            correct_nature += (preds_nature == labels_nature).sum().item()
            correct_materiality += (preds_materiality == labels_materiality).sum().item()
            correct_biological += (preds_biological == labels_biological).sum().item()
            correct_landscape += (preds_landscape == labels_landscape).sum().item()
            total += images.size(0)
    
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    print("Accuracy por tarea:")
    print(f"  Nature: {correct_nature/total*100:.2f}%")
    print(f"  Materiality: {correct_materiality/total*100:.2f}%")
    print(f"  Biological: {correct_biological/total*100:.2f}%")
    print(f"  Landscape: {correct_landscape/total*100:.2f}%")'''

    
    model.eval()

    for img in range(30):
        # Select one sample from the test set.
        sample_image, sample_image_name, sample_labels = test_dataset[img]
        # Unsqueeze to add batch dimension.
        input_tensor = sample_image.unsqueeze(0).to(device)
        # For ResNet50, we register hooks on the layer4 of the underlying resnet.
        target_layer = model.backbone.backbone.layer4
        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate_cam(input_tensor)  # cam is a numpy array of shape (H, W)
        # Resize cam to 224x224 (input image size)
        cam_resized = cv2.resize(cam, (224, 224))
        # Prepare the original image for overlay (denormalize)
        original_img = sample_image.cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        original_img = std * original_img + mean
        original_img = np.clip(original_img, 0, 1)
        # Plot the original image and overlay the heatmap.
        plt.imshow(original_img)
        plt.imshow(cam_resized, cmap='jet', alpha=0.5)
        plt.title("GradCAM - Nature Visual")
        plt.axis('off')

        gradcam_path = os.path.join(output_dir, "grad_cam")
        os.makedirs(gradcam_path, exist_ok=True)

        gradcam_output_path = os.path.join(gradcam_path, f"{sample_image_name}_gradcam.png")
        plt.savefig(gradcam_output_path, bbox_inches='tight')
        plt.close()
        print(f"\nGradCAM output saved to {gradcam_output_path}")
    # ---------------------------------------------------------------------



    test_loss = 0.0
    total = 0

    all_preds_nature, all_labels_nature = [], []
    all_preds_materiality, all_labels_materiality = [], []
    all_preds_biological, all_labels_biological = [], []
    all_preds_landscape, all_labels_landscape = [], []
    
    with torch.no_grad():
        for images, image_names, labels in test_loader:
            images = images.to(device)
            labels_nature = labels['nature_visual'].to(device)
            labels_materiality = labels['nep_materiality_visual'].to(device)
            labels_biological = labels['nep_biological_visual'].to(device)
            labels_landscape = labels['landscape-type_visual'].to(device)
            
            out_nature, out_materiality, out_biological, out_landscape = model(images)
            loss_nature = criterion_nature(out_nature, labels_nature)
            loss_materiality = criterion_materiality(out_materiality, labels_materiality)
            loss_biological = criterion_biological(out_biological, labels_biological)
            loss_landscape = criterion_landscape(out_landscape, labels_landscape)
            loss = loss_nature + loss_materiality + loss_biological + loss_landscape
            test_loss += loss.item()
            
            preds_nature = out_nature.argmax(dim=1)
            preds_materiality = out_materiality.argmax(dim=1)
            preds_biological = out_biological.argmax(dim=1)
            preds_landscape = out_landscape.argmax(dim=1)
            
            all_preds_nature.extend(preds_nature.cpu().numpy())
            all_labels_nature.extend(labels_nature.cpu().numpy())
            all_preds_materiality.extend(preds_materiality.cpu().numpy())
            all_labels_materiality.extend(labels_materiality.cpu().numpy())
            all_preds_biological.extend(preds_biological.cpu().numpy())
            all_labels_biological.extend(labels_biological.cpu().numpy())
            all_preds_landscape.extend(preds_landscape.cpu().numpy())
            all_labels_landscape.extend(labels_landscape.cpu().numpy())
            
            total += images.size(0)
    
    avg_test_loss = test_loss / len(test_loader)
    print(f"\nTest Loss: {avg_test_loss:.4f}")
    
    # Print accuracy per task
    accuracy_nature = np.mean(np.array(all_preds_nature) == np.array(all_labels_nature))
    accuracy_materiality = np.mean(np.array(all_preds_materiality) == np.array(all_labels_materiality))
    accuracy_biological = np.mean(np.array(all_preds_biological) == np.array(all_labels_biological))
    accuracy_landscape = np.mean(np.array(all_preds_landscape) == np.array(all_labels_landscape))
    
    print("\nAccuracy por tarea:")
    print(f"  Nature: {accuracy_nature*100:.2f}%")
    print(f"  Materiality: {accuracy_materiality*100:.2f}%")
    print(f"  Biological: {accuracy_biological*100:.2f}%")
    print(f"  Landscape: {accuracy_landscape*100:.2f}%")
    
    # Compute and print confusion matrices and classification reports
    print("\nConfusion Matrix - Nature:")
    print(confusion_matrix(all_labels_nature, all_preds_nature))
    print("Classification Report - Nature:")
    print(classification_report(all_labels_nature, all_preds_nature))
    
    print("\nConfusion Matrix - Materiality:")
    print(confusion_matrix(all_labels_materiality, all_preds_materiality))
    print("Classification Report - Materiality:")
    print(classification_report(all_labels_materiality, all_preds_materiality))
    
    print("\nConfusion Matrix - Biological:")
    print(confusion_matrix(all_labels_biological, all_preds_biological))
    print("Classification Report - Biological:")
    print(classification_report(all_labels_biological, all_preds_biological))
    
    print("\nConfusion Matrix - Landscape:")
    print(confusion_matrix(all_labels_landscape, all_preds_landscape))
    print("Classification Report - Landscape:")
    print(classification_report(all_labels_landscape, all_preds_landscape))


    # ---------------------------------------------------------------------------------------


    output_test_dir = os.path.join(output_dir, "test_predictions")
    os.makedirs(output_test_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for images, image_names, labels in test_loader:
            images = images.to(device)
            out_nature, out_materiality, out_biological, out_landscape = model(images)
            
            preds_nature = out_nature.argmax(dim=1)
            preds_materiality = out_materiality.argmax(dim=1)
            preds_biological = out_biological.argmax(dim=1)
            preds_landscape = out_landscape.argmax(dim=1)
            
            for i in range(len(images)):
                gt_nature = labels['nature_visual'][i]
                gt_materiality = labels['nep_materiality_visual'][i]
                gt_biological = labels['nep_biological_visual'][i]
                gt_landscape = labels['landscape-type_visual'][i]
                
                pred_nature = preds_nature[i].item()
                pred_materiality = preds_materiality[i].item()
                pred_biological = preds_biological[i].item()
                pred_landscape = preds_landscape[i].item()
                
                title = (f"Name: {image_names[i]}\n"
                        f"GT - Nature: {gt_nature}, Materiality: {gt_materiality}, "
                        f"Biological: {gt_biological}, Landscape: {gt_landscape}\n"
                        f"Pred - Nature: {pred_nature}, Materiality: {pred_materiality}, "
                        f"Biological: {pred_biological}, Landscape: {pred_landscape}")
                
                save_path = os.path.join(output_test_dir, f"{image_names[i]}_pred.png")
                imshow_tensor(images[i], title=title, save_path=save_path)
                
            break

    
if __name__ == '__main__':
    main()
