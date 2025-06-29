import os
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

'''
Normalize all of the values in the dataset between [0, 1]
'''
def normalize_data(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    denom = np.where(max_val - min_val == 0, 1e-8, max_val - min_val)
    return (data - min_val) / denom

def gen_chart(norm_data, decompressed_data, labels, dataset, method, detector, cr):
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(norm_data[:, 0], label='Original', linewidth=1)
    ax.plot(decompressed_data[:, 0], label='Decompressed', linestyle='--', linewidth=1)

    anomaly_indices = np.where(labels == 1)[0]

    ax.set_title(f'{dataset} - CR: {cr} - Method: {method.name}')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.legend()
    fig.tight_layout()

    plot_out_dir = os.path.join("charts", method.name, detector)
    os.makedirs(plot_out_dir, exist_ok=True)
    plot_path = os.path.join(plot_out_dir, f"{dataset.split('.')[0]}_cr{cr}.png")

    try:
        fig.savefig(plot_path)
        print(f"Saved plot to {plot_path}")
    except Exception as e:
        print(f"Error saving plot {plot_path}: {e}")
    finally:
        plt.close(fig)
