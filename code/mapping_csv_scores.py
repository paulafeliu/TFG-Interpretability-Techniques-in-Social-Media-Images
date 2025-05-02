import pandas as pd
import ast

def map_prediction_string(cell, mapping):
    """
    Given a cell value that is either a dict or a string representation of a dict,
    parse it, map the 'predicted_label' using the provided mapping, and return the updated dict.
    """
    try:
        # If the cell is a string, convert it into a dict.
        if isinstance(cell, str):
            d = ast.literal_eval(cell)
        else:
            d = cell

        # Ensure the cell has the 'predicted_label' key, then map the number to its label.
        if "predicted_label" in d:
            original_label = d["predicted_label"]
            d["predicted_label"] = mapping.get(original_label, original_label)
        return d
    except Exception as e:
        print(f"Error processing cell {cell}: {e}")
        return cell

def main():

    csv_path = '/fhome/pfeliu/tfg_feliu/TFG-Interpretability-Techniques-in-Social-Media-Images/output/preds_spanish_df_scores.csv'  
    output_csv = '/fhome/pfeliu/tfg_feliu/TFG-Interpretability-Techniques-in-Social-Media-Images/output/preds_spanish_df_scores(mapped).csv' 

    # Define your label maps from string to number.
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

    # Create inverse mapping: number -> string.
    inverse_label_maps = {
        col: {num: label for label, num in mapping.items()} 
        for col, mapping in label_maps.items()
    }

    df = pd.read_csv(csv_path)

    # Rename the first column to "image_filename" if needed.
    first_col = df.columns[0]
    df.rename(columns={first_col: "image_filename"}, inplace=True)

    print("CSV cargado:")
    print(df.head())

    # For every column where you want to apply the mapping (if it exists in the CSV)
    for col, mapping in inverse_label_maps.items():
        if col in df.columns:
            # Convert each cell (which is a dict stored as a string) with our mapping function.
            df[col] = df[col].apply(lambda x: map_prediction_string(x, mapping))
        else:
            print(f"Advertencia: La columna '{col}' no se encontró en el CSV.")

    print("\nCSV luego del mapping:")
    print(df.head())

    df.to_csv(output_csv, index=False)
    print(f"\nCSV modificado guardado en: {output_csv}")

if __name__ == '__main__':
    main()
