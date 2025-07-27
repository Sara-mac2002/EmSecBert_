import requests
import pandas as pd
import os
import time
from bs4 import BeautifulSoup
from datetime import datetime
import re

def fetch_mitre_enterprise_data():
    """Fetch the MITRE ATT&CK Enterprise framework data from the API"""
    print("Fetching MITRE ATT&CK Enterprise data from API...")
    url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
    
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}")

def extract_techniques_from_api(stix_data):
    """Extract techniques and subtechniques from the STIX data"""
    techniques = []
    
    for obj in stix_data.get('objects', []):
        if (obj.get('type') == 'attack-pattern' and 
            obj.get('x_mitre_domains') and 
            'enterprise-attack' in obj.get('x_mitre_domains')):
            
            # Get technique ID (e.g., T1548 or T1548.001 for subtechniques)
            technique_id = "N/A"
            technique_url = "N/A"
            for ref in obj.get('external_references', []):
                if ref.get('source_name') == 'mitre-attack':
                    technique_id = ref.get('external_id', 'N/A')
                    technique_url = ref.get('url', 'N/A')
                    break
            
            # Determine if this is a technique or subtechnique
            is_subtechnique = '.' in technique_id
            
            technique = {
                'ID': technique_id,
                'Name': obj.get('name', 'N/A'),
                'Type': 'Subtechnique' if is_subtechnique else 'Technique',
                'Description': obj.get('description', 'N/A'),
                'URL': technique_url
            }
            
            techniques.append(technique)
    
    return techniques

def scrape_technique_details(technique):
    """Scrape additional details for a technique from the MITRE ATT&CK website, focusing on mitigations"""
    if technique['URL'] == 'N/A':
        return {
            'Mitigations': 'N/A'
        }
    
    print(f"Scraping details for {technique['ID']} - {technique['Name']}...")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(technique['URL'], headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch {technique['URL']}: Status code {response.status_code}")
            return {
                'Mitigations': 'N/A'
            }
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # ------- EXTRACT MITIGATIONS -------
        mitigations = []
        
        # Look for mitigations table - this is the most reliable method
        mitigation_tables = soup.find_all('table')
        mitigation_table = None
        
        for table in mitigation_tables:
            # Check if table header contains "ID" and "Mitigation"
            headers = table.find_all('th')
            header_texts = [h.get_text(strip=True) for h in headers]
            if 'ID' in header_texts and any('Mitigation' in h for h in header_texts):
                mitigation_table = table
                break
        
        if mitigation_table:
            rows = mitigation_table.find_all('tr')[1:]  # Skip header row
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 2:
                    mitigation_id = cells[0].get_text(strip=True)
                    mitigation_name = cells[1].get_text(strip=True)
                    # Format exactly as requested: M0807: Network Allowlists
                    mitigations.append(f"{mitigation_id}: {mitigation_name}")
        
        # If table approach didn't work, try looking for mitigation sections
        if not mitigations:
            mitigation_headers = soup.find_all(['h2', 'h3', 'h4'], string=lambda t: t and 'Mitigation' in t)
            for header in mitigation_headers:
                mitigation_section = header.find_next_sibling(['div', 'ul'])
                if mitigation_section:
                    for item in mitigation_section.find_all(['li', 'div']):
                        mitigation_text = item.get_text(strip=True)
                        # Look for mitigation ID pattern (M0000, M0001, etc.)
                        match = re.search(r'(M\d{4})(?:\s*[:,-]\s*)([^,;.]+)', mitigation_text)
                        if match:
                            mitigation_id = match.group(1)
                            mitigation_name = match.group(2).strip()
                            mitigations.append(f"{mitigation_id}: {mitigation_name}")
                        elif mitigation_text and mitigation_text not in mitigations:
                            # If no ID found, just use the text as is
                            mitigations.append(mitigation_text)
        
        return {
            'Mitigations': ", ".join(mitigations) if mitigations else 'N/A'
        }
        
    except Exception as e:
        print(f"Error scraping {technique['URL']}: {str(e)}")
        return {
            'Mitigations': 'N/A'
        }

def enhance_techniques_with_web_data(techniques, delay=1):
    """Enhance technique data with information scraped from the web"""
    enhanced_techniques = []
    
    for i, technique in enumerate(techniques):
        # Add a delay to avoid overloading the server
        if i > 0:
            time.sleep(delay)
        
        # Get additional details from website
        details = scrape_technique_details(technique)
        
        # Combine API data with scraped data
        enhanced_technique = dict(technique)
        enhanced_technique['Mitigations'] = details['Mitigations']
        
        enhanced_techniques.append(enhanced_technique)
        
        # Show progress
        print(f"Processed {i+1}/{len(techniques)} techniques")
    
    return enhanced_techniques

def save_to_excel(data, file_path):
    """Save data to Excel file"""
    df = pd.DataFrame(data)
    
    # Reorder columns for better readability
    column_order = [
        'ID', 'Name', 'Type', 'Description', 'Mitigations'
    ]
    
    # Only include columns that exist
    existing_columns = [col for col in column_order if col in df.columns]
    remaining_columns = [col for col in df.columns if col not in column_order]
    df = df[existing_columns + remaining_columns]
    
    # Export to Excel
    df.to_excel(file_path, index=False)
    print(f"Data successfully exported to {file_path}")

def main():
    # Create output directory
    output_dir = "mitre_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get technique info from API
    stix_data = fetch_mitre_enterprise_data()
    techniques = extract_techniques_from_api(stix_data)
    print(f"Found {len(techniques)} techniques from API")
    
    # For testing with a small subset, uncomment these lines:
    # test_techniques = techniques[:5]  # Just process first 5 techniques
    # enhanced_techniques = enhance_techniques_with_web_data(test_techniques)
    
    # Process all techniques
    enhanced_techniques = enhance_techniques_with_web_data(techniques)
    
    # Save results
    output_file = os.path.join(output_dir, "MITRE_ATT&CK_Enterprise.xlsx")  
    save_to_excel(enhanced_techniques, output_file)
    
    # Also save as CSV for backup
    csv_file = os.path.join(output_dir, "MITRE_ATT&CK_Enterprise.csv")
    pd.DataFrame(enhanced_techniques).to_csv(csv_file, index=False)
    print(f"Data also saved to {csv_file}")
    
    print("\nExtraction completed successfully!")
    print(f"Data has been saved to {output_file} and {csv_file}")
    print("The extracted data includes:")
    print("- Technique/Subtechnique ID")
    print("- Name")
    print("- Type (Technique or Subtechnique)")
    print("- Description")
    print("- Mitigations (in the format 'M0807: Network Allowlists, M0810: Out-of-Band Communications Channel')")

if __name__ == "__main__":
    main()