import pandas as pd
import re
import os
import numpy as np
#  Replace "your_project_directory_path" with your actual project directory path

# Use the file path you provided
file_path = "your_project_directory_path/Data_Extraction/MITRE_ICS/MITRE_ATT&CK_ICS_Enhanced.xlsx"

# Ensure the file exists
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
else:
    # Load your existing dataset
    df = pd.read_excel(file_path)
    
    # Create a new dataframe for the transformed structure
    transformed_rows = []
    
    # Process each technique
    for _, row in df.iterrows():
        technique_id = row['ID'] if 'ID' in row else 'N/A'
        technique_name = row['Name'] if 'Name' in row else 'N/A'
        technique_description = row['Description'] if 'Description' in row else 'N/A'
        tactics = row['Tactics'] if 'Tactics' in row else 'N/A'
        
        # Handle mitigations, ensure it's a string
        if 'Mitigations' in row:
            mitigations = str(row['Mitigations']) if not pd.isna(row['Mitigations']) else 'N/A'
        else:
            mitigations = 'N/A'
        
        # Handle assets, ensure it's a string and not NaN
        if 'Assets' in row and not pd.isna(row['Assets']):
            assets = str(row['Assets'])
            
            # Check if assets is not 'N/A' or empty
            if assets != 'N/A' and assets.strip():
                # Split multiple assets
                asset_list = assets.split(', ')
                
                for asset in asset_list:
                    # Try to extract asset ID and name
                    match = re.match(r'(A\d+):\s*(.*)', asset)
                    if match:
                        asset_id = match.group(1)
                        asset_name = match.group(2)
                    else:
                        asset_id = 'Unknown'
                        asset_name = asset
                    
                    # Create a row for this technique-target pair
                    transformed_rows.append({
                        'Target_ID': asset_id,
                        'Target_Name': asset_name,
                        'Technique_ID': technique_id,
                        'Technique_Name': technique_name,
                        'Technique_Description': technique_description,
                        'Tactics': tactics,
                        'Security_Controls': mitigations
                    })
            else:
                # Add row with no specific target
                transformed_rows.append({
                    'Target_ID': 'N/A',
                    'Target_Name': 'N/A',
                    'Technique_ID': technique_id,
                    'Technique_Name': technique_name,
                    'Technique_Description': technique_description,
                    'Tactics': tactics,
                    'Security_Controls': mitigations
                })
        else:
            # Add row with no specific target
            transformed_rows.append({
                'Target_ID': 'N/A',
                'Target_Name': 'N/A',
                'Technique_ID': technique_id,
                'Technique_Name': technique_name,
                'Technique_Description': technique_description,
                'Tactics': tactics,
                'Security_Controls': mitigations
            })
    
    # Create the new dataframe
    transformed_df = pd.DataFrame(transformed_rows)
    
    # Organize columns for clarity
    column_order = [
        'Target_ID', 'Target_Name', 
        'Technique_ID', 'Technique_Name', 'Technique_Description', 'Tactics', 
        'Security_Controls'
    ]
    
    # Only include columns that exist
    existing_columns = [col for col in column_order if col in transformed_df.columns]
    transformed_df = transformed_df[existing_columns]
    
    # Save to Excel
    output_path = os.path.join(os.path.dirname(file_path), "MITRE_ATT&CK_ICS_Target_Technique_Controls2.xlsx")
    transformed_df.to_excel(output_path, index=False)
    
    print(f"Transformation complete! Output saved to: {output_path}")
    print(f"Created {len(transformed_df)} rows from {len(df)} original techniques")