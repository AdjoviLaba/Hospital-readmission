
import pandas as pd
import os
import argparse
import json
import ast

def parse_args():
    parser = argparse.ArgumentParser(description='Check and fix CSV files for hospital readmission prediction')
    parser.add_argument('--data_dir', type=str, default='./processed_data/',
                        help='Directory containing train.csv and test.csv')
    parser.add_argument('--fix', action='store_true', 
                        help='Fix issues found in CSV files')
    return parser.parse_args()

def check_column_names(df):
    """Check if the dataframe has the required columns"""
    required_columns = ['ID', 'TEXT', 'Label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        print(f"Available columns: {df.columns.tolist()}")
        return False
    return True

def check_sentence_attribute(df):
    """Check if sentence_attribute column can be parsed as a list"""
    if 'sentence_attribute' not in df.columns:
        print("No 'sentence_attribute' column found - this is needed for the full model but not for pure_bert mode")
        return False
    
    issues = []
    for idx, row in df.iterrows():
        try:
            sentence_attr = ast.literal_eval(row['sentence_attribute'])
            if not isinstance(sentence_attr, list):
                issues.append(idx)
        except (SyntaxError, ValueError):
            issues.append(idx)
    
    if issues:
        print(f"Found issues with sentence_attribute in {len(issues)} rows")
        print(f"Example problematic rows: {issues[:5]}")
        return False
    return True

def fix_for_pure_bert(df):
    """Create a minimal dataframe suitable for pure_bert mode"""
    # For pure_bert mode, we only need ID, TEXT, and Label columns
    if 'ID' not in df.columns:
        # Create an ID column if it doesn't exist
        df['ID'] = range(len(df))
    
    if 'TEXT' not in df.columns and 'note_text' in df.columns:
        # Rename note_text to TEXT if necessary
        df['TEXT'] = df['note_text']
    
    if 'Label' not in df.columns and 'readmission' in df.columns:
        # Rename readmission to Label if necessary
        df['Label'] = df['readmission']
    
    # Keep only the required columns
    columns_to_keep = ['ID', 'TEXT', 'Label']
    df = df[columns_to_keep]
    
    # Add empty columns needed for compatibility with the code
    df['sentence_attribute'] = df.apply(lambda x: [], axis=1)
    df['MT'] = df.apply(lambda x: [], axis=1)
    df['dependency'] = df.apply(lambda x: [], axis=1)
    
    return df

def main():
    args = parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist")
        return
    
    # Check train.csv
    train_path = os.path.join(args.data_dir, 'train.csv')
    if not os.path.exists(train_path):
        print(f"Error: {train_path} does not exist")
    else:
        print(f"Checking {train_path}...")
        try:
            train_df = pd.read_csv(train_path)
            print(f"Successfully loaded train.csv with {len(train_df)} rows")
            
            valid_columns = check_column_names(train_df)
            if args.fix and not valid_columns:
                print("Fixing train.csv for pure_bert mode...")
                fixed_df = fix_for_pure_bert(train_df)
                backup_path = train_path + '.backup'
                if not os.path.exists(backup_path):
                    train_df.to_csv(backup_path, index=False)
                    print(f"Backed up original to {backup_path}")
                fixed_df.to_csv(train_path, index=False)
                print(f"Fixed {train_path}")
        
        except Exception as e:
            print(f"Error loading {train_path}: {str(e)}")
    
    # Check test.csv
    test_path = os.path.join(args.data_dir, 'test.csv')
    if not os.path.exists(test_path):
        print(f"Error: {test_path} does not exist")
    else:
        print(f"Checking {test_path}...")
        try:
            test_df = pd.read_csv(test_path)
            print(f"Successfully loaded test.csv with {len(test_df)} rows")
            
            valid_columns = check_column_names(test_df)
            if args.fix and not valid_columns:
                print("Fixing test.csv for pure_bert mode...")
                fixed_df = fix_for_pure_bert(test_df)
                backup_path = test_path + '.backup'
                if not os.path.exists(backup_path):
                    test_df.to_csv(backup_path, index=False)
                    print(f"Backed up original to {backup_path}")
                fixed_df.to_csv(test_path, index=False)
                print(f"Fixed {test_path}")
        
        except Exception as e:
            print(f"Error loading {test_path}: {str(e)}")

def create_sample_data(output_dir):
    """Create sample train.csv and test.csv files for testing"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create minimal sample data
    train_data = {
        'ID': [1, 2, 3, 4],
        'TEXT': [
            "Patient shows signs of improved respiratory function after treatment.",
            "Persistent fever and cough, no response to antibiotics.",
            "Blood pressure stabilized, medication adjusted.",
            "Post-surgical wound healing well, no signs of infection."
        ],
        'Label': [0, 1, 0, 0]  # 0 = no readmission, 1 = readmission
    }
    
    test_data = {
        'ID': [5, 6],
        'TEXT': [
            "Discharge with normal vital signs, follow-up in two weeks.",
            "Patient reports chest pain and shortness of breath."
        ],
        'Label': [0, 1]
    }
    
    # Add empty columns needed for the code
    for data in [train_data, test_data]:
        data['sentence_attribute'] = [[] for _ in range(len(data['ID']))]
        data['MT'] = [[] for _ in range(len(data['ID']))]
        data['dependency'] = [[] for _ in range(len(data['ID']))]
    
    # Convert to dataframes and save
    pd.DataFrame(train_data).to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    pd.DataFrame(test_data).to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"Created sample data files in {output_dir}")

if __name__ == "__main__":
    main()