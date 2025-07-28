# EMsecBERT: A Fine-tuned BERT-based LLM for Automated Cybersecurity Threat Analysis of Electric Mobility
This repository provides the first open-source fine-tuned BERT-based Large Language Model specifically designed for automated cybersecurity threat analysis in electric mobility infrastructure. This work presents a complete methodology from dataset creation to model deployment for Named Entity Recognition in cybersecurity threat descriptions.
## Overview
Electric mobility infrastructure integrates information and communication technologies to enhance operational efficiency, but this integration also introduces significant cybersecurity vulnerabilities. This research addresses the challenge through:
- **Dataset Construction** : Semi-automated creation of annotated cybersecurity datasets
- **Model Development**: Fine-tuning multiple BERT variants for cybersecurity NER
- **Performance Evaluation**: Comprehensive comparison across transformer architectures
- **Production Integration**: Deployment-ready implementation for threat analysis
# Research Methodology
## 1. Data Sources and Collection
We drew from authoritative cybersecurity knowledge bases:
- **MITRE ATT&CK Enterprise**: 63 entries covering IT infrastructure threats
- **MITRE ATT&CK ICS**: 95 entries focusing on industrial control systems
- **Sandia National Laboratories Report**: 18 entries on electric mobility security
## 2. Dataset Construction Pipeline
```python
# Historical data collection (completed during research)
# Available in Data_Extraction/ for reference
python Data_Extraction/MITRE_ICS/MITRE_ATT&CK_ICS.py
python Data_Extraction/MITRE_ENTREPRISE/MITRE_ENTREPRISE.py
```
## 3. Annotation Strategy
- **Manual Annotation**: 16 entries manually annotated by cybersecurity experts (4,281 tokens)
- **LLM-Based Expansion**: Gemini 2.5 Pro used to scale to 176 entries (49,726 tokens)
- **Inter-annotator Agreement**: κ = 0.941 (almost perfect agreement)
- **BIO Tagging Scheme**: B-X (beginning), I-X (inside), O (outside) for three entity types
## 4. Entity Categories
- **TARGET_ASSET**: Systems, devices, or components targeted by threats
- **PRECON**: Preconditions or prerequisites for attacks
- **MITIGATION**: Security measures and countermeasures


### Installation
```python
# Clone the repository
git clone https://github.com/yourusername/EMsecBERT.git
cd EMsecBERT

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
Model Fine-tuning and Evaluation
Fine-tuning Multiple BERT Variants
# Fine-tune BERT baseline
python Automated_Extraction/fine-tune/bert.py --dataset_path Automated_Extraction/Datasets/

# Fine-tune domain-adapted models
python Automated_Extraction/fine-tune/cysecbert.py --epochs 10 --batch_size 16
python Automated_Extraction/fine-tune/secbert.py --learning_rate 2e-5
python Automated_Extraction/fine-tune/securebert.py --dataset_path Automated_Extraction/Datasets/
```

### Model Architecture
All models use BERT+CRF architecture:
- **Contextual Encoder**: Pre-trained BERT for token representations
- **CRF Layer**: Ensures valid BIO tag sequences
- **Fine-tuning**: End-to-end fine-tuning on cybersecurity datasets
```python

## Using the Pre-trained EMsecBERT Model
# Basic Entity Extraction


from emobility.automated.extraction import EMsecBERTExtractor 
# Initialize with best-performing model checkpoint
 extractor = EMsecBERTExtractor( model_checkpoint_path="Automated_Extraction/model/CySecBert_crf_checkpoint.pt" )
 # Extract entities from threat description 
threat_text = """Adversaries may inject malicious code into the charging station controller if the device firmware is outdated. Implement certificate-based authentication to prevent unauthorized access.""" entities = extractor.extract_entities(threat_text) 
print(f"Target Assets: {entities.get('TARGET_ASSET', [])}") 
print(f"Preconditions: {entities.get('PRECON', [])}")
 print(f"Mitigations: {entities.get('MITIGATION', [])}")
```


### Dataset Structure
Training Data Organization
Automated_Extraction/Datasets/
├── Mitre_ENTREPRISE/           # Enterprise IT threats (63 entries)
│   ├── train.txt              # Training split (70%)
│   ├── valid.txt              # Validation split (15%)
│   ├── test.txt               # Test split (15%)
│   └── BIO.txt                # Complete BIO-tagged dataset
├── Mitre_ICS/                 # Industrial Control Systems (95 entries)
│   ├── train.txt              # Training split
│   ├── valid.txt              # Validation split
│   └── test.txt                  # Test split
│   └── BIO.txt                # Complete BIO-tagged dataset
└── Paper_report/              # Sandia report data (18 entries)
    ├── train.txt              # Training split
    ├── valid.txt              # Validation split
    └── test.txt               # Test split
    └── BIO.txt                # Complete BIO-tagged dataset

### Dataset Statistics
* Total Entries: 176 threat scenarios
* Total Tokens: 49,726 labeled tokens
* Label Distribution:
    * TARGET_ASSET: 2,613 B-tags, 1,405 I-tags
    * PRECON: 299 B-tags, 411 I-tags
    * MITIGATION: 49 B-tags, 372 I-tags
* Data Split: 70% train, 15% validation, 15% test
Command Line Interface


## 5. Result for EmSecBert available in Automated_Extraction/model/results.txt
Evaluation Metrics
All models evaluated using:
* Token-level accuracy: Proportion of correctly predicted tokens
* Entity-level precision: Correctly predicted entities / Total predicted entities
* Entity-level recall: Correctly predicted entities / Total actual entities
* F1-score: Harmonic mean of precision and recall

### Integration with EMOBILITY-DSML Framework
```python

from emobility.automated.extraction import EMsecBERTExtractor

# Initialize with best-performing model checkpoint
extractor = EMsecBERTExtractor(
    model_checkpoint_path="Automated_Extraction/model/CySecBert_crf_checkpoint.pt"
)

# Extract entities from threat description
from emobility.automated.extraction import EMsecBERTExtractor
#  Replace "your_project_directory_path" with your actual project directory path          
# Initialize extractor (requires fine-tuned model checkpoint)
extractor = EMsecBERTExtractor(
    model_checkpoint_path="your_project_directory_path/Automated_Extraction/model/CySecBert_crf_checkpoint.pt"
)

# Process ICS data and extract entities
ics_results = extractor.process_mitre_data(ics_data, "ICS_Extracted_Entities")

# Process Enterprise data and extract entities
enterprise_results = extractor.process_mitre_data(enterprise_data, "Enterprise_Extracted_Entities")

# Process both datasets
for framework, data in both_data.items():
    results = extractor.process_mitre_data(data, f"{framework.upper()}_Extracted_Entities")
    print(f"Processed {len(results)} entries for {framework}")
```

## Project Structure
EMsecBERT/
├── Automated_Extraction/           # Core research implementation
│   ├── Datasets/                   # Training datasets (176 entries, 49,726 tokens)
│   ├── fine-tune/                  # Model training scripts
│   ├── model/                      # Best checkpoint (CySecBERT F1: 81.44%)
│   └── requirements/               # Dependencies
├── Data_Extraction/               # Historical data collection scripts(research phase)
│   ├── MITRE_ENTREPRISE/          # Enterprise data extraction
│   └── MITRE_ICS/                 # ICS data extraction
├── emobility/automated/extraction/
   ├──__init__.py                 # Package initialization
   ├── __main__.py                 # CLI entry point
   ├── mitre_collector.py          # MITRE ATT&CK data collection
   ├── EMsecBERT_extractor.py     # Entity extraction with BERT
   ├── requirements.txt            # Dependencies
   ├── setup.py                   # Package setup
   └── README.md                  # This file


Requirements
* Python 3.8+
* PyTorch 1.9+
* Transformers 4.0+
* Pandas 1.3+
* scikit-learn 1.0+
See requirements.txt for complete dependency list.

Acknowledgments
* MITRE Corporation for the ATT&CK framework
* Sandia National Laboratories for electric mobility security research
* Google for Gemini 2.5 Pro used in dataset annotation
* Open-source community for transformer models and tools
