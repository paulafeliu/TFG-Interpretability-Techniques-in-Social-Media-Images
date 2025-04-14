import pandas as pd
import numpy as np

def main():

    csv_path = '/fhome/pfeliu/tfg_feliu/TFG-Interpretability-Techniques-in-Social-Media-Images/output/preds_spanish_dataset.csv'  
    output_csv = '/fhome/pfeliu/tfg_feliu/TFG-Interpretability-Techniques-in-Social-Media-Images/output/preds_spanish_dataset(mapped).csv' 

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

    inverse_label_maps = {col: {num: label for label, num in mapping.items()} 
                      for col, mapping in label_maps.items()}

    df = pd.read_csv(csv_path)

    first_col = df.columns[0]
    df.rename(columns={first_col: "image_filename"}, inplace=True)

    print("CSV cargado:")
    print(df.head())

    for col, mapping in inverse_label_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
        else:
            print(f"Advertencia: La columna '{col}' no se encontró en el CSV.")

    print("\nCSV luego del mapping:")
    print(df.head())

    df.to_csv(output_csv, index=False)
    print(f"\nCSV modificado guardado en: {output_csv}")

if __name__ == '__main__':
    main()
