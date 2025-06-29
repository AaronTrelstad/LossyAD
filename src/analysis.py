import os
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
