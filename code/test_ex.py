#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
import copy

# Configuración de Pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

def process_csv(file_path):
    df = pd.read_csv(file_path)
    new_rows = []
    # Iteramos sobre los índices (0 a 3) para generar las filas procesadas
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
        new_rows.append(temp_df)
    
    new_df = pd.concat(new_rows, ignore_index=True)
    new_df = new_df.dropna(subset=['nep_materiality_visual', 'nep_biological_visual'])
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
    
    new_order = ['image_name', 'nature_visual', 'nep_materiality_visual', 'nep_biological_visual', 
                 'landscape-type_visual', 'platform_id', 'index']
    new_df = new_df[new_order]
    
    def tiene_contradicciones(valor):
        if pd.isna(valor):
            return False
        items = [item.strip() for item in str(valor).split(';')]
        return len(set(items)) > 1

    cols = ['nature_visual', 'nep_materiality_visual', 'nep_biological_visual', 'landscape-type_visual']
    df_bool = new_df[cols].applymap(tiene_contradicciones)
    agreed_df = new_df[~df_bool.any(axis=1)]
    
    return agreed_df

class MultiTaskImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, label_maps=None):
        self.data = pd.read_csv(csv_file).fillna("NaN")
        self.img_dir = img_dir
        self.transform = transform
        self.label_maps = label_maps

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        base_img_name = row['image_name']
        extensions = ['.jpg', '.png', '.jpeg']
        img_path = None
        for ext in extensions:
            candidate = os.path.join(self.img_dir, base_img_name + ext).replace('\\', '/')
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            raise FileNotFoundError(f"No se encontró la imagen: {base_img_name} con ninguna de las extensiones {extensions}")
        
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

# Nuevo dataset para imágenes sin etiquetas (test)
class UnlabeledImageDataset(Dataset):
    def __init__(self, img_dir, image_files, transform=None):
        self.img_dir = img_dir
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.img_dir, filename).replace('\\','/')
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # El id se obtiene del nombre de archivo sin extensión
        base_name = os.path.splitext(filename)[0]
        return image, base_name

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

class ResNet50Backbone(nn.Module):
    def __init__(self, backbone):
        super(ResNet50Backbone, self).__init__()
        self.backbone = backbone
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return x

class MultiTaskModel(nn.Module):
    def __init__(self, backbone, feature_dim=2048):
        super(MultiTaskModel, self).__init__()
        self.backbone = backbone
        self.fc_nature = nn.Linear(feature_dim, 2)
        self.fc_materiality = nn.Linear(feature_dim, 2)
        self.fc_biological = nn.Linear(feature_dim, 2)
        self.fc_landscape = nn.Linear(feature_dim, 8)
        
    def forward(self, x):
        features = self.backbone(x)
        out_nature = self.fc_nature(features)
        out_materiality = self.fc_materiality(features)
        out_biological = self.fc_biological(features)
        out_landscape = self.fc_landscape(features)
        return out_nature, out_materiality, out_biological, out_landscape

def main():
    #print("Starting")
    # Rutas de los CSV y directorios (ajusta estas rutas según tu entorno)
    csv_path1 = '/fhome/pfeliu/tfg_feliu/data/39_20250401_0816.csv'
    csv_path2 = '/fhome/pfeliu/tfg_feliu/data/43.csv'
    csv_path3 = '/fhome/pfeliu/tfg_feliu/data/43_1.csv'
    
    df1 = process_csv(csv_path1)
    df2 = process_csv(csv_path2)
    df3 = process_csv(csv_path3)
    #print("CSV processed")
    
    # Unimos los CSV y filtramos algunas filas
    agree_df = pd.concat([df1, df2, df3], ignore_index=True)
    agree_df = agree_df[~agree_df["landscape-type_visual"].isin(["forest_and_seminatural_areas,water_bodies", "artificial_surfaces,water_bodies"])]
    agree_df = agree_df.drop_duplicates()
    #print("agreed_df created") 

    agreed_df_path = '/fhome/pfeliu/tfg_feliu/data/X_labels_agreements_0204.csv'
    agree_df.to_csv(agreed_df_path, index=False)
    #print("agreed_df saved")

    # Definición de transformaciones
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Definición del mapeo de etiquetas
    label_maps = {
        "nature_visual": {"Yes": 1, "No": 0},
        "nep_materiality_visual": {"material": 0, "immaterial": 1},
        "nep_biological_visual": {"biotic": 0, "abiotic": 1},
        "landscape-type_visual": {
            "artificial_surfaces": 0,
            "forest_and_seminatural_areas": 1,
            "wetlands": 2,
            "water_bodies": 3,
            "agricultural_areas": 4,
            "other": 5,
            "none": 6,
            "NaN": 7
        }
    }
    
    # Imprime información de las etiquetas
    for col in ['nature_visual', 'nep_materiality_visual', 'nep_biological_visual', 'landscape-type_visual']:
        print(f"Etiquetas únicas en {col}:")
        print(agree_df[col].unique())
        print("------")
        print(agree_df[col].value_counts())
    
    # --- CREACIÓN DEL DATASET DE ENTRENAMIENTO ---
    # Usamos todo el CSV etiquetado para entrenar
    img_dir = '/fhome/pfeliu/tfg_feliu/data/twitter'
    train_dataset = MultiTaskImageDataset(csv_file=agreed_df_path, img_dir=img_dir, transform=transform, label_maps=label_maps)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print("Total de imágenes en train:", len(train_dataset))
    
    # --- CREACIÓN DEL DATASET DE TEST ---
    # Se seleccionan las imágenes de la carpeta de twitter que NO están en el CSV (según su id)
    all_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    labeled_ids = set(agree_df['image_name'])
    test_files = [f for f in all_files if os.path.splitext(f)[0] not in labeled_ids]
    print("Total de imágenes en test (sin etiquetas):", len(test_files))
    test_files = test_files[:50]
    
    test_dataset = UnlabeledImageDataset(img_dir=img_dir, image_files=test_files, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    output_dir = "/fhome/pfeliu/tfg_feliu/data/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Directorio para guardar imágenes con predicciones del test
    test_preds_dir = os.path.join(output_dir, "test_preds_uncoded")
    os.makedirs(test_preds_dir, exist_ok=True)
    
    # Configurar el modelo (puedes ajustar el uso de pesos según versión)
    resnet = models.resnet50(pretrained=True)
    resnet.fc = nn.Identity()
    backbone_model = ResNet50Backbone(resnet)
    model = MultiTaskModel(backbone_model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # --- ENTRENAMIENTO ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    num_epochs = 50
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
            loss_nature = criterion(out_nature, labels_nature)
            loss_materiality = criterion(out_materiality, labels_materiality)
            loss_biological = criterion(out_biological, labels_biological)
            loss_landscape = criterion(out_landscape, labels_landscape)
            loss = loss_nature + loss_materiality + loss_biological + loss_landscape
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")
    
    # --- PREDICCIÓN SOBRE LAS IMÁGENES TEST ---
    model.eval()
    # Creamos un diccionario inverso para mapear números a etiquetas
    rev_label_maps = {}
    for task, mapping in label_maps.items():
        rev_label_maps[task] = {v: k for k, v in mapping.items()}
    
    with torch.no_grad():
        for images, base_names in test_loader:
            images = images.to(device)
            out_nature, out_materiality, out_biological, out_landscape = model(images)
            preds_nature = out_nature.argmax(dim=1)
            preds_materiality = out_materiality.argmax(dim=1)
            preds_biological = out_biological.argmax(dim=1)
            preds_landscape = out_landscape.argmax(dim=1)
            
            for i in range(len(images)):
                pred_nature = rev_label_maps['nature_visual'].get(preds_nature[i].item(), str(preds_nature[i].item()))
                pred_materiality = rev_label_maps['nep_materiality_visual'].get(preds_materiality[i].item(), str(preds_materiality[i].item()))
                pred_biological = rev_label_maps['nep_biological_visual'].get(preds_biological[i].item(), str(preds_biological[i].item()))
                pred_landscape = rev_label_maps['landscape-type_visual'].get(preds_landscape[i].item(), str(preds_landscape[i].item()))
                
                title = (f"ID: {base_names[i]}\n"
                         f"Pred - Nature: {pred_nature}, Materiality: {pred_materiality}, "
                         f"Biological: {pred_biological}, Landscape: {pred_landscape}")
                save_path = os.path.join(test_preds_dir, f"{base_names[i]}_pred.png")
                # Importante: usamos .cpu() para llevar la imagen a CPU antes de guardarla
                imshow_tensor(images[i].cpu(), title=title, save_path=save_path)
    
    print("Predicciones en test guardadas en:", test_preds_dir)

if __name__ == '__main__':
    main()