import os
import pandas as pd
import matplotlib.pyplot as plt
import csv
from concurrent.futures import ThreadPoolExecutor


def calc_aggregate(detector, args):
    for compression_method in os.listdir(args.results_dir):
        ad_method_path = os.path.join(args.results_dir, compression_method)

        if not os.path.isdir(ad_method_path):
            continue

        for ad_method in os.listdir(ad_method_path):
            cr_file_path = os.path.join(ad_method_path, ad_method)
            if not os.path.isdir(cr_file_path):
                continue

            print(f"Processing: {cr_file_path}")
            summary_rows = []

            for file in os.listdir(cr_file_path):
                if not file.endswith(".csv") or file == "summary.csv":
                    continue

                compression_ratio = os.path.splitext(file)[0] 
                file_path = os.path.join(cr_file_path, file)

                try:
                    df = pd.read_csv(file_path)
                    if df.empty:
                        continue

                    avg_metrics = df.drop(columns=["Dataset"]).mean()

                    avg_metrics["compression_ratio"] = compression_ratio
                    summary_rows.append(avg_metrics)

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue

            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                columns = ["compression_ratio"] + [col for col in summary_df.columns if col != "compression_ratio"]
                summary_df = summary_df[columns]
                summary_df.to_csv(os.path.join(cr_file_path, "summary.csv"), index=False)
                print(f"Saved summary to {os.path.join(cr_file_path, 'summary.csv')}")

def plot_aggregate(args):
    for compression_method in os.listdir(args.results_dir):
        ad_method_path = os.path.join(args.results_dir, compression_method)

        if not os.path.isdir(ad_method_path):
            continue

        for ad_method in os.listdir(ad_method_path):
            cr_file_path = os.path.join(ad_method_path, ad_method)
            if not os.path.isdir(cr_file_path):
                continue

            print(f"Processing: {cr_file_path}")

            for file in os.listdir(cr_file_path):
                if file != "summary.csv":
                    continue

                full_path = os.path.join(cr_file_path, file)

                x = []
                y = []

                with open(full_path, 'r') as csvfile:
                    data = csv.reader(csvfile, delimiter=",")

                    for i, row in enumerate(data):
                        if i == 0:
                            continue 
                        if len(row) < 7:  
                            print(f"Skipping malformed row {i}: {row}")
                            continue
                        try:
                            x_val = float(row[0])
                            y_val = float(row[6])
                        except ValueError:
                            print(f"Skipping non-numeric row {i}: {row}")
                            continue
                        
                        x.append(x_val)
                        y.append(y_val)

                if not x or not y:
                    print("No valid data found to plot.")
                else:
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.scatter(x, y, label='Original', s=15)

                    ax.set_xlabel('Compression Ratio')
                    ax.set_ylabel('F1')
                    ax.legend()
                    fig.tight_layout()

                    plot_out_dir = os.path.join("charts", ad_method, compression_method)
                    os.makedirs(plot_out_dir, exist_ok=True)
                    plot_path = os.path.join(plot_out_dir, f"test_{ad_method}_{compression_method}.png")

                    try:
                        fig.savefig(plot_path)
                        print(f"Saved plot to {plot_path}")
                    except Exception as e:
                        print(f"Error saving plot {plot_path}: {e}")
                    finally:
                        plt.close(fig)


def run_analysis(args):
    def calc_aggregate_worker(method):
        calc_aggregate(method, args)

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(
            calc_aggregate_worker,
            detector
        )
        for detector in args.ad_methods]

        for future in futures:
            future.result()
    
    plot_aggregate(args)

