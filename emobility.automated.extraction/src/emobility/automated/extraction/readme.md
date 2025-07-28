# EMOBILITY DSML - Automated Extraction Module

This module provides automated extraction capabilities for cybersecurity frameworks, specifically focused on MITRE ATT&CK data collection and entity extraction using EMsecBERT for electric mobility security analysis.

## Features

- **MITRE ATT&CK Data Collection**: Automated collection from ICS and Enterprise frameworks
- **Entity Extraction**: Fine-tuned BERT model for extracting cybersecurity entities
- **Multiple Output Formats**: Export to CSV and Excel
- **Command Line Interface**: Easy-to-use CLI for batch processing

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### 1. Collect MITRE ATT&CK Data

```python
from emobility.automated.extraction import MitreCollector

# Initialize collector
collector = MitreCollector(output_dir="data/mitre")

# Collect ICS data
ics_data = collector.collect_mitre_ics_data()

# Collect Enterprise data
enterprise_data = collector.collect_mitre_enterprise_data()

# Collect both frameworks
both_data = {
    'ics': collector.collect_mitre_ics_data(),
    'enterprise': collector.collect_mitre_enterprise_data()
}
```

### 2. Extract Entities with EMsecBERT

```python
from emobility.automated.extraction import EMsecBERTExtractor
#  Replace "your_project_directory_path" with your actual project directory path          
# Initialize extractor (requires fine-tuned model checkpoint)
extractor = EMsecBERTExtractor(
    model_checkpoint_path="your_project_directory_path/Automated_Extraction/model/CySecBert_crf_checkpoint.pt"
)

# Process MITRE data and extract entities
results = extractor.process_mitre_data(ics_data, "ICS_Extracted_Entities")
```

## Command Line Usage

### Collect MITRE Data

```bash
# Collect ICS framework data
python -m emobility.automated.extraction --collect mitre --framework ics

# Collect Enterprise framework data
python -m emobility.automated.extraction --collect mitre --framework enterprise

# Collect both frameworks
python -m emobility.automated.extraction --collect mitre --framework both
```

### Extract Entities

```bash
# Extract entities from collected data
python -m emobility.automated.extraction --extract entities --input data/mitre_data.csv
```

### Full Pipeline

```bash
# Run complete pipeline: collect + extract
python -m emobility.automated.extraction --pipeline --framework both
```

## Entity Types Extracted

The EMsecBERT model extracts three types of cybersecurity entities:

- **Target Assets**: Systems, devices, or components that are targeted
- **Preconditions**: Requirements or conditions needed for an attack
- **Mitigations**: Security measures or countermeasures

## Output Format

The extraction produces files with the following columns:

| Column | Description |
|--------|-------------|
| Name | MITRE technique name |
| Description | Original technique description |
| Extracted_Target_Assets | Comma-separated list of identified assets |
| Extracted_Preconditions | Comma-separated list of identified preconditions |
| Extracted_Mitigations | Comma-separated list of identified mitigations |

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.0+
- Pandas 1.3+
- BeautifulSoup4 4.9+

See `requirements.txt` for complete dependency list.

## Model Checkpoint

The EMsecBERT extractor requires a fine-tuned model checkpoint. Make sure to:

1. Have the checkpoint file available
2. Update the path in your code or CLI arguments

## Project Structure

```
emobility/automated/extraction/
├── __init__.py                 # Package initialization
├── __main__.py                 # CLI entry point
├── mitre_collector.py          # MITRE ATT&CK data collection
├── EMsecBERT_extractor.py     # Entity extraction with BERT
├── requirements.txt            # Dependencies
├── setup.py                   # Package setup
└── README.md                  # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

