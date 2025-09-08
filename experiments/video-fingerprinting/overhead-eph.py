import pandas as pd
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    df = pd.read_csv(csv_file)

    # In every dataset name, remove "-seed0", where 0 goes from 0 to 99
    df['dataset'] = df['dataset'].str.replace(r"-seed\d+", "", regex=True)

    # in the delay column, print top-10 values
    print("Top-10 delay values:")
    print(df["delay"].sort_values(ascending=False).head(10))

    # remove the top-two delays
    df = df.drop(df.nlargest(2, 'delay').index)

    # Group by 'dataset' and compute mean and std for 'load' and 'delay'
    summary = df.groupby("dataset")[["load", "delay"]].agg(['mean', 'std'])

    # Flatten MultiIndex columns
    summary.columns = ['_'.join(col) for col in summary.columns]
    summary.reset_index(inplace=True)

    # Format and print LaTeX output
    for _, row in summary.iterrows():
        dataset = row["dataset"]
        load_mean = round(row["load_mean"] * 100, 1)
        load_std = round(row["load_std"] * 100, 1)
        delay_mean = round(row["delay_mean"] * 100, 1)
        delay_std = round(row["delay_std"] * 100, 1)

        latex_load = f"${load_mean}^{{\\pm {load_std}}}$"
        latex_delay = f"${delay_mean}^{{\\pm {delay_std}}}$"
        print(f"{dataset} & {latex_load} & {latex_delay} \\\\")

if __name__ == "__main__":
    main()

