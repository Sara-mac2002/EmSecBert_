import pandas as pd
import os
from datetime import datetime

def extract_descriptions(input_file, output_file):
    """
    Extract only the descriptions from a MITRE ATT&CK CSV file
    and save them to a new CSV file with only descriptions.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to the output CSV file
    """
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' does not exist.")
        return
    
    # Check if it's a CSV file
    if not input_file.lower().endswith('.csv'):
        print(f"Error: File '{input_file}' is not a CSV file.")
        return
    
    try:
        # Read the CSV file
        print(f"Reading data from '{input_file}'...")
        df = pd.read_csv(input_file)
        
        # Check if 'Description' column exists
        if 'Description' not in df.columns:
            print("Error: 'Description' column not found in the CSV file.")
            return
        
        # Extract only the descriptions
        descriptions_df = pd.DataFrame({'Description': df['Description']})
        
        # Save to CSV
        descriptions_df.to_csv(output_file, index=False)
        print(f"Success! Descriptions extracted to: '{output_file}'")
        print(f"Total descriptions extracted: {len(descriptions_df)}")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
#  Replace "your_project_directory_path" with your actual project directory path
# Hardcoded input and output paths
input_file = "your_project_directory_path/Data_Extraction/MITRE_ENTREPRISE/MITRE_ATT&CK_Enterprise.csv"
output_file = "your_project_directory_path/Data_Extraction/MITRE_ENTREPRISE/MITRE_ATT&CK_Enterprise_DESCRIPTION.csv"

# Run the extraction
if __name__ == "__main__":
    print("Starting extraction process...")
    extract_descriptions(input_file, output_file)
    print("Extraction process completed.")
