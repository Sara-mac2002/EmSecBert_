#!/usr/bin/env python3
"""
EMOBILITY DSML Automated Extraction - Main Entry Point

This module allows the package to be run as:
python -m emobility.automated.extraction

Provides a command-line interface for both MITRE data collection
and EMsecBERT entity extraction.
"""
import os 
import sys
import argparse
from .mitre_collector import MitreCollector
from .EMsecBERT_extractor import EMsecBERTExtractor


def main():
    """Main entry point for the extraction module"""
    parser = argparse.ArgumentParser(
        description="EMOBILITY DSML Automated Extraction Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect MITRE ATT&CK data
  python -m emobility.automated.extraction --collect mitre --framework ics
  python -m emobility.automated.extraction --collect mitre --framework enterprise
  python -m emobility.automated.extraction --collect mitre --framework both
  
  # Extract entities from collected data
  python -m emobility.automated.extraction --extract entities --input data.csv
  
  # Full pipeline: collect and extract
  python -m emobility.automated.extraction --pipeline --framework both
        """
    )
    
    # Main operation modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--collect', 
        choices=['mitre'], 
        help='Collect data from specified source'
    )
    group.add_argument(
        '--extract', 
        choices=['entities'], 
        help='Extract entities from text data'
    )
    group.add_argument(
        '--pipeline', 
        action='store_true', 
        help='Run full pipeline: collect + extract'
    )
    
    # Framework selection for MITRE collection
    parser.add_argument(
        '--framework', 
        choices=['ics', 'enterprise', 'both'], 
        default='ics',
        help='MITRE framework to collect (default: ics)'
    )
    
    # Input/Output options
    parser.add_argument(
        '--input', 
        type=str, 
        help='Input file path for extraction'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='data/mitre',
        help='Output directory (default: data/mitre)'
    )
 #  Replace "your_project_directory_path" with your actual project directory path   
    parser.add_argument(
        '--model-path', 
        type=str, 
        default='your_project_directory_path/Automated_Extraction/model/CySecBert_crf_checkpoint.pt',
        help='Path to EMsecBERT model checkpoint'
    )
    args = parser.parse_args()
    
    try:
        if args.collect == 'mitre':
            collect_mitre_data(args)
        elif args.extract == 'entities':
            extract_entities(args)
        elif args.pipeline:
            run_full_pipeline(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def collect_mitre_data(args):
    """Collect MITRE ATT&CK data"""
    print("=== MITRE ATT&CK Data Collection ===")
    
    collector = MitreCollector(output_dir=args.output_dir)
    
    if args.framework == 'ics':
        print("Collecting ICS framework data...")
        collector.collect_mitre_ics_data()
    elif args.framework == 'enterprise':
        print("Collecting Enterprise framework data...")
        collector.collect_mitre_enterprise_data()
    elif args.framework == 'both':
        print("Collecting both ICS and Enterprise framework data...")
        collector.collect_mitre_ics_data()
        collector.collect_mitre_enterprise_data()
    
    print("✓ MITRE data collection completed!")


def extract_entities(args):
    """Extract entities using EMsecBERT"""
    print("=== Entity Extraction with EMsecBERT ===")
    
    if not args.input:
        print("Error: --input file is required for entity extraction")
        sys.exit(1)
    
    extractor = EMsecBERTExtractor(model_checkpoint_path=args.model_path)
    
    # Load data from file and process
    import pandas as pd
    try:
        if args.input.endswith('.csv'):
            data = pd.read_csv(args.input)
        elif args.input.endswith('.xlsx'):
            data = pd.read_excel(args.input)
        else:
            print("Error: Input file must be CSV or Excel format")
            sys.exit(1)
        
        # Convert to required format
        mitre_data = data[['Name', 'Description']].to_dict('records')
        
        # Extract entities
        output_base = args.input.rsplit('.', 1)[0] + "_extracted"
        results = extractor.process_mitre_data(mitre_data, output_base)
        
        print(f"✓ Entity extraction completed! Processed {len(results)} techniques.")
        
    except Exception as e:
        print(f"Error processing input file: {e}")
        sys.exit(1)


def run_full_pipeline(args):
    """Run the complete pipeline: collect + extract"""
    print("=== Full Pipeline: Collect + Extract ===")
    
    # Step 1: Collect MITRE data
    collect_mitre_data(args)
    
    # Step 2: Extract entities from collected data
    print("\n=== Starting Entity Extraction ===")
    
    extractor = EMsecBERTExtractor(model_checkpoint_path=args.model_path)
    
    if args.framework in ['ics', 'both']:
        try:
            # Find the ICS file
            import os
            import glob
            
            ics_files = glob.glob(os.path.join(args.output_dir, "*ICS*.csv"))
            if ics_files:
                latest_ics = max(ics_files, key=os.path.getctime)
                print(f"Processing ICS data from: {latest_ics}")
                
                import pandas as pd
                data = pd.read_csv(latest_ics)
                mitre_data = data[['Name', 'Description']].to_dict('records')
                
                output_base = os.path.join(args.output_dir, "ICS_Extracted_Entities")
                extractor.process_mitre_data(mitre_data, output_base)
                print("✓ ICS entity extraction completed!")
        except Exception as e:
            print(f"Warning: Could not process ICS data: {e}")
    
    if args.framework in ['enterprise', 'both']:
        try:
            # Find the Enterprise file
            enterprise_files = glob.glob(os.path.join(args.output_dir, "*Enterprise*.csv"))
            if enterprise_files:
                latest_enterprise = max(enterprise_files, key=os.path.getctime)
                print(f"Processing Enterprise data from: {latest_enterprise}")
                
                import pandas as pd
                data = pd.read_csv(latest_enterprise)
                mitre_data = data[['Name', 'Description']].to_dict('records')
                
                output_base = os.path.join(args.output_dir, "Enterprise_Extracted_Entities")
                extractor.process_mitre_data(mitre_data, output_base)
                print("✓ Enterprise entity extraction completed!")
        except Exception as e:
            print(f"Warning: Could not process Enterprise data: {e}")
    
    print("\n✓ Full pipeline completed successfully!")


if __name__ == "__main__":
    main()