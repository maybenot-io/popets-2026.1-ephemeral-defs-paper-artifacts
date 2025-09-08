import pandas as pd
import sys
import re

if len(sys.argv) != 2:
    print("Usage: python script.py <filename>")
    sys.exit(1)

filename = sys.argv[1]

# Function to extract pc, ps, and window values from the dataset string
def extract_values(dataset):
    match = re.search(r'pc(?P<pc>\d+\.\d+)-ps(?P<ps>\d+\.\d+)-w(?P<w>[\d,]+)', dataset)
    if match:
        w_values = match.group('w').split(',')
        return match.group('pc', 'ps') + tuple(w_values)
    return [None] * 3

try:
    # Read CSV file
    df = pd.read_csv(filename)
    
    # Select relevant columns
    selected_columns = ['dataset', 'missing', 'bandwidth', 'delay', 'rf', 'df']
    df_selected = df[selected_columns].copy()
    
    # Extract pc, ps, w values from the dataset column
    extracted_values = df_selected['dataset'].apply(lambda x: extract_values(x))
    max_w_count = max(len(x) - 2 for x in extracted_values)  # Get max number of w values
    w_columns = [f'w{i+1}' for i in range(max_w_count)]
    columns = ['pc', 'ps'] + w_columns
    
    df_extracted = pd.DataFrame(extracted_values.tolist(), columns=columns)
    
    # Round values
    df_selected['missing'] = df_selected['missing'].round(4) # we want to see very small values
    df_selected['bandwidth'] = df_selected['bandwidth'].round(2)
    df_selected['delay'] = df_selected['delay'].round(2)
    df_selected['rf'] = df_selected['rf'].round(2)
    df_selected['df'] = df_selected['df'].round(2)
    
    # Concatenate extracted values with bandwidth and delay columns
    df_final = pd.concat([df_extracted, df_selected[['missing', 'bandwidth', 'delay', 'rf', 'df']]], axis=1)
    
    # Overhead, 2x weight for delay
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

