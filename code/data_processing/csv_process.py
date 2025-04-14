"""
csv_processor.py
----------------
Module for processing CSV files and returning a cleaned DataFrame.
"""

import os
import numpy as np
import pandas as pd


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

        # Filter rows whose image exists in the directory
        def image_exists(row):
            base_img_name = row['image_name']
            for ext in valid_extensions:
                img_path = os.path.join(img_dir, base_img_name + ext)
                if os.path.exists(img_path):
                    return True
            return False
        
        temp_df = temp_df[temp_df.apply(image_exists, axis=1)]
        new_rows.append(temp_df)

    new_df = pd.concat(new_rows, ignore_index=True)
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
        if len(items) == 1 or len(set(items)) == 1:
            return items[0]
        return "; ".join(items)
    
    new_df['nature_visual'] = new_df['nature_visual'].apply(process_nature)
    new_df['nep_materiality_visual'] = new_df['nep_materiality_visual'].apply(process_other)
    new_df['nep_biological_visual'] = new_df['nep_biological_visual'].apply(process_other)
    new_df['landscape-type_visual'] = new_df['landscape-type_visual'].apply(process_other)

    # Apply mask to filter out inconsistent rows
    mask = (new_df['nature_visual'] == "No") | (
        (new_df['nature_visual'] == "Yes") &
        new_df['nep_materiality_visual'].notna() &
        new_df['nep_biological_visual'].notna()
    )
    masked_df = new_df[mask]

    new_order = ['image_name', 'nature_visual', 'nep_materiality_visual', 
                 'nep_biological_visual', 'landscape-type_visual', 'platform_id', 'index']
    masked_df = masked_df[new_order]
    
    def tiene_contradicciones(valor):
        if pd.isna(valor):
            return False
        items = [item.strip() for item in str(valor).split(';')]
        return len(set(items)) > 1

    cols = ['nep_materiality_visual', 'nep_biological_visual', 'landscape-type_visual']
    df_bool = masked_df[cols].applymap(tiene_contradicciones)
    agreed_df = masked_df[~df_bool.any(axis=1)]


    print("El DataFrame tiene", len(agreed_df), "filas.")
    
    return agreed_df
