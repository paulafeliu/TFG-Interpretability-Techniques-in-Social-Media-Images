import os
import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import denormalize_image  # asegúrate de tener esta función para visualizar correctamente las imágenes


def shap_analysis(model, test_dataset, device, output_dir, num_images=10):
    print("\nStarting SHAP analysis...")
    model.eval()
    model.to(device)

    '''shap_output_dir = os.path.join(output_dir, "shap")
    os.makedirs(shap_output_dir, exist_ok=True)'''

    # Preparamos background (Deep SHAP requiere background para calcular valores esperados)
    print("Preparing background samples for SHAP...")
    background_samples = [test_dataset[i][0] for i in np.random.choice(len(test_dataset), 50, replace=False)]
    background_tensor = torch.stack(background_samples).to(device)

    # Wrapper del modelo para SHAP
    def model_forward(x):
        with torch.no_grad():
            output = model(x.to(device))
            if isinstance(output, (list, tuple)):
                output = torch.cat([o.float() for o in output], dim=1)
            return output

    print("Initializing SHAP DeepExplainer...")
    explainer = shap.DeepExplainer(model_forward, background_tensor)

    for i in tqdm(range(num_images), desc="Generating SHAP explanations"):
        sample_image, sample_image_name, sample_label = test_dataset[i]
        input_tensor = sample_image.unsqueeze(0).to(device)

        # Calcular valores SHAP
        shap_values = explainer.shap_values(input_tensor)

        # Convertir a formato numpy para visualización
        original_image = denormalize_image(sample_image)  # (H, W, C), ya en rango 0-1
        original_image = np.clip(original_image, 0, 1)
        input_np_batch = np.expand_dims(original_image, axis=0)  # (1, H, W, C)

        # === Visualización SHAP ===
        try:
            image_plot_path = os.path.join(shap_output_dir, f"{sample_image_name}_shap_image_plot.png")
            shap.image_plot(shap_values, input_np_batch, show=False)
            plt.savefig(image_plot_path, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"[WARNING] Error in shap.image_plot: {e}")

        # === Visualización Bar plot para cada tarea (si multitarea) ===
        if isinstance(shap_values, list):
            for task_idx, task_shap in enumerate(shap_values):
                try:
                    bar_plot_path = os.path.join(shap_output_dir, f"{sample_image_name}_task{task_idx}_bar.png")
                    shap.summary_plot(task_shap, input_tensor.cpu().numpy(), show=False)
                    plt.savefig(bar_plot_path, bbox_inches="tight")
                    plt.close()
                except Exception as e:
                    print(f"[WARNING] Could not create bar plot for task {task_idx}: {e}")
        else:
            try:
                bar_plot_path = os.path.join(shap_output_dir, f"{sample_image_name}_bar.png")
                shap.summary_plot(shap_values, input_tensor.cpu().numpy(), show=False)
                plt.savefig(bar_plot_path, bbox_inches="tight")
                plt.close()
            except Exception as e:
                print(f"[WARNING] Could not create bar plot: {e}")

        print(f"SHAP visualizations saved for {sample_image_name}")

    print("SHAP analysis complete.\n")
