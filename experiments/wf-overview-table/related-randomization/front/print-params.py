import pandas as pd
import sys
import re

if len(sys.argv) != 2:
    print("Usage: python script.py <filename>")
    sys.exit(1)

filename = sys.argv[1]

# Function to extract budget, wmin, wmax, and states values from the dataset string
def extract_values(dataset):
    match = re.search(r'budget(?P<budget>\d+)-wmin(?P<wmin>\d+)-wmax(?P<wmax>\d+)-states(?P<states>\d+)', dataset)
    if match:
        return match.group('budget', 'wmin', 'wmax', 'states')
    return [None] * 4

try:
    # Read CSV file
    df = pd.read_csv(filename)
    
    # Select relevant columns
    selected_columns = ['dataset', 'bandwidth', 'delay', 'rf', 'df']
    df_selected = df[selected_columns].copy()
    
    # Extract budget, wmin, wmax, states values from the dataset column
    extracted_values = df_selected['dataset'].apply(lambda x: extract_values(x))
    columns = ['budget', 'wmin', 'wmax', 'states']
    
    df_extracted = pd.DataFrame(extracted_values.tolist(), columns=columns)
    
    # Round to 2 decimal places
    df_selected['bandwidth'] = df_selected['bandwidth'].round(2)
    df_selected['delay'] = df_selected['delay'].round(2)
    df_selected['rf'] = df_selected['rf'].round(2)
    df_selected['df'] = df_selected['df'].round(2)
    
    # Concatenate extracted values with bandwidth and time columns
    df_final = pd.concat([df_extracted, df_selected[['bandwidth', 'delay', 'rf', 'df']]], axis=1)

    # Sum, bandwidth and delay with weight 2.0
    df_final['overhead'] = df_final['delay'].astype('float')*2.0 + df_final['bandwidth'].astype('float')
    df_final['attack'] = df_final['df'].astype('float') + df_final['rf'].astype('float')

    print("\n\n############################## Sorted by Attack (DF+RF Accuracy) ##############################")
    df_final = df_final.sort_values('attack')
    print(df_final.to_markdown(index=False))
    
    print("\n\n############################## Sorted by Overhead (Bandwidth + 2*Delay) ##############################")
    df_final = df_final.sort_values('overhead')
    print(df_final.to_markdown(index=False))
    

except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    sys.exit(1)
except KeyError as e:
    print(f"Error: Missing required column {e}")
    sys.exit(1)

