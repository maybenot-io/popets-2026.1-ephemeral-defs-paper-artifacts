import pandas as pd
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    df = pd.read_csv(csv_file)

    # Group by 'dataset' and compute mean and std for 'rf' and 'df'
    summary = df.groupby("dataset")[["rf", "df"]].agg(['mean', 'std'])

    # Flatten MultiIndex columns
    summary.columns = ['_'.join(col) for col in summary.columns]
    summary.reset_index(inplace=True)

    # Format and print LaTeX output
    for _, row in summary.iterrows():
        dataset = row["dataset"]
        rf_mean = round(row["rf_mean"] * 100, 1)
        rf_std = round(row["rf_std"] * 100, 1)
        df_mean = round(row["df_mean"] * 100, 1)
        df_std = round(row["df_std"] * 100, 1)

        latex_rf = f"${rf_mean}^{{\\pm {rf_std}}}$"
        latex_df = f"${df_mean}^{{\\pm {df_std}}}$"
        print(f"{dataset} & {latex_df} & {latex_rf} \\\\")

if __name__ == "__main__":
    main()

