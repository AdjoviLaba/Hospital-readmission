#!/usr/bin/env python3
"""
MIMIC-IV Custom Data Preparation Script

This script adapts to your specific MIMIC-IV dataset structure, using transfers.csv 
and ICU stays to create readmission prediction data.

Usage:
    python prepare_mimic_iv_custom.py --mimic_path /path/to/mimic-iv --output_path ./processed_data
"""

import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime, timedelta
import re
from tqdm import tqdm
import gc

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare MIMIC-IV data for readmission prediction')
    
    parser.add_argument('--mimic_path', type=str, required=True,
                        help='Path to the MIMIC-IV dataset directory')
    parser.add_argument('--output_path', type=str, default='./processed_data',
                        help='Output directory for processed data')
    parser.add_argument('--window', type=int, default=30,
                        help='Readmission window in days (default: 30)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for train/test split')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    
    return parser.parse_args()

def extract_data_from_transfers(mimic_path, window=30):
    """Extract readmission data from MIMIC-IV using transfers and ICU stays"""
    print("Loading MIMIC-IV data...")
    
    # Define file paths
    transfers_path = os.path.join(mimic_path, "hosp/transfers.csv")
    patients_path = os.path.join(mimic_path, "hosp/patients.csv")
    icustays_path = os.path.join(mimic_path, "icu/icustays.csv")
    procedures_path = os.path.join(mimic_path, "hosp/procedures_icd.csv")
    
    # Check if files exist
    if not os.path.exists(transfers_path):
        raise FileNotFoundError(f"Could not find transfers file at {transfers_path}")
    if not os.path.exists(patients_path):
        raise FileNotFoundError(f"Could not find patients file at {patients_path}")
    
    # Load transfers data
    print(f"Reading transfers data from {transfers_path}")
    transfers = pd.read_csv(transfers_path)
    
    # Load patients data
    print(f"Reading patients data from {patients_path}")
    patients = pd.read_csv(patients_path)
    
    # Load ICU stays if available
    icu_stays = None
    if os.path.exists(icustays_path):
        print(f"Reading ICU stays from {icustays_path}")
        icu_stays = pd.read_csv(icustays_path)
    
    # Process transfers to identify hospital stays
    print("Processing transfers to identify hospital stays...")
    
    # Convert datetime columns
    transfers['intime'] = pd.to_datetime(transfers['intime'])
    transfers['outtime'] = pd.to_datetime(transfers['outtime'])
    
    # Group transfers by hadm_id to find admission and discharge times
    hospital_stays = transfers.groupby('hadm_id').agg({
        'subject_id': 'first',
        'intime': 'min',
        'outtime': 'max'
    }).reset_index()
    
    # Rename columns for clarity
    hospital_stays.rename(columns={'intime': 'admittime', 'outtime': 'dischtime'}, inplace=True)
    
    # Sort by patient and admission time
    hospital_stays = hospital_stays.sort_values(['subject_id', 'admittime'])
    
    # Calculate time to next admission for each patient
    hospital_stays['next_admittime'] = hospital_stays.groupby('subject_id')['admittime'].shift(-1)
    hospital_stays['days_to_next_admit'] = (
        (hospital_stays['next_admittime'] - hospital_stays['dischtime']).dt.total_seconds() / (24 * 3600)
    )
    
    # Create readmission label (1 if readmitted within window, 0 otherwise)
    hospital_stays['Label'] = np.where(
        (hospital_stays['days_to_next_admit'] <= window) & 
        (hospital_stays['days_to_next_admit'] >= 0), 
        1, 0
    )
    
    # Merge with patient demographics
    hospital_stays = hospital_stays.merge(patients, on='subject_id', how='left')
    
    # Add ICU stays information if available
    if icu_stays is not None:
        # Count ICU stays per hospital admission
        icu_counts = icu_stays.groupby('hadm_id').size().reset_index(name='icu_stays_count')
        hospital_stays = hospital_stays.merge(icu_counts, on='hadm_id', how='left')
        hospital_stays['icu_stays_count'] = hospital_stays['icu_stays_count'].fillna(0)
        
        # Get ICU length of stay
        icu_stays['los'] = pd.to_numeric(icu_stays['los'], errors='coerce')
        icu_los = icu_stays.groupby('hadm_id')['los'].sum().reset_index(name='total_icu_los')
        hospital_stays = hospital_stays.merge(icu_los, on='hadm_id', how='left')
        hospital_stays['total_icu_los'] = hospital_stays['total_icu_los'].fillna(0)
    
    # Add procedures if available
    if os.path.exists(procedures_path):
        print(f"Reading procedures from {procedures_path}")
        procedures = pd.read_csv(procedures_path)
        
        # Count procedures per hospital admission
        proc_counts = procedures.groupby('hadm_id').size().reset_index(name='procedure_count')
        hospital_stays = hospital_stays.merge(proc_counts, on='hadm_id', how='left')
        hospital_stays['procedure_count'] = hospital_stays['procedure_count'].fillna(0)
    
    # Calculate length of stay
    hospital_stays['length_of_stay'] = (
        (hospital_stays['dischtime'] - hospital_stays['admittime']).dt.total_seconds() / (24 * 3600)
    )
    
    # Create a synthetic TEXT field with combined information
    print("Creating synthetic clinical text from structured data...")
    
    def create_synthetic_text(row):
        text_parts = []
        text_parts.append(f"Hospital Admission ID: {row['hadm_id']}")
        text_parts.append(f"Gender: {row['gender']}")
        
        if 'anchor_age' in row:
            text_parts.append(f"Age: {row['anchor_age']}")
        
        text_parts.append(f"Length of Stay: {row['length_of_stay']:.1f} days")
        
        if 'icu_stays_count' in row:
            text_parts.append(f"Number of ICU Stays: {int(row['icu_stays_count'])}")
        
        if 'total_icu_los' in row and row['total_icu_los'] > 0:
            text_parts.append(f"Total ICU Length of Stay: {row['total_icu_los']:.1f} days")
        
        if 'procedure_count' in row:
            text_parts.append(f"Number of Procedures: {int(row['procedure_count'])}")
        
        return "\n".join(text_parts)
    
    tqdm.pandas(desc="Creating synthetic clinical text")
    hospital_stays['TEXT'] = hospital_stays.progress_apply(create_synthetic_text, axis=1)
    
    # Select relevant columns
    data = hospital_stays[['hadm_id', 'subject_id', 'TEXT', 'Label']]
    data.rename(columns={'hadm_id': 'ID'}, inplace=True)
    
    print(f"Created dataset with {len(data)} hospital stays and readmission labels")
    return data

def add_lab_data(data, mimic_path):
    """Add lab events data if available"""
    print("Checking for lab data...")
    
    lab_path = os.path.join(mimic_path, "hosp/labevents.csv")
    if not os.path.exists(lab_path):
        print("Lab events file not found, skipping lab data")
        return data
    
    print("Processing lab events (this may take a while)...")
    
    # Since lab data can be very large, we'll process it in chunks
    chunk_size = 100000
    lab_counts = pd.DataFrame()
    
    for chunk in tqdm(pd.read_csv(lab_path, chunksize=chunk_size), desc="Processing lab chunks"):
        # Get count of lab tests per admission
        chunk_counts = chunk.groupby('hadm_id').size().reset_index(name='temp_count')
        
        if lab_counts.empty:
            lab_counts = chunk_counts
        else:
            # Merge with existing counts
            lab_counts = lab_counts.merge(chunk_counts, on='hadm_id', how='outer', suffixes=('', '_new'))
            lab_counts['temp_count'] = lab_counts['temp_count'].fillna(0) + lab_counts['temp_count_new'].fillna(0)
            lab_counts = lab_counts[['hadm_id', 'temp_count']]
    
    # Rename column
    lab_counts.rename(columns={'temp_count': 'lab_count'}, inplace=True)
    
    # Merge with main data
    data_with_id = data.rename(columns={'ID': 'hadm_id'})
    data_with_id = data_with_id.merge(lab_counts, on='hadm_id', how='left')
    
    # Update TEXT with lab information
    def add_lab_info(row):
        text = row['TEXT']
        if pd.notna(row['lab_count']):
            text += f"\nNumber of Lab Tests: {int(row['lab_count'])}"
        return text
    
    tqdm.pandas(desc="Adding lab information to text")
    data_with_id['TEXT'] = data_with_id.progress_apply(add_lab_info, axis=1)
    
    # Restore original column names
    data_with_id.rename(columns={'hadm_id': 'ID'}, inplace=True)
    
    return data_with_id[['ID', 'subject_id', 'TEXT', 'Label']]

def split_data(data, test_size=0.2, seed=42):
    """Split data into train and test sets by patient ID"""
    print("Splitting data into train and test sets...")
    
    # Get unique patient IDs
    patients = data['subject_id'].unique()
    
    # Set random seed
    np.random.seed(seed)
    
    # Randomly select patients for testing
    test_size_patients = int(len(patients) * test_size)
    test_patients = np.random.choice(patients, size=test_size_patients, replace=False)
    
    # Split data
    train_data = data[~data['subject_id'].isin(test_patients)].copy()
    test_data = data[data['subject_id'].isin(test_patients)].copy()
    
    # Remove subject_id column
    train_data = train_data.drop('subject_id', axis=1)
    test_data = test_data.drop('subject_id', axis=1)
    
    print(f"Train set: {len(train_data)} records from {len(patients) - test_size_patients} patients")
    print(f"Test set: {len(test_data)} records from {test_size_patients} patients")
    
    return train_data, test_data

def process_for_concept_extraction(train_data, test_data):
    """Add empty concept extraction columns for compatibility with the model"""
    print("Adding empty concept columns for model compatibility...")
    
    for dataset in [train_data, test_data]:
        # Add required columns with empty values for compatibility
        dataset['MT'] = dataset.apply(lambda x: [], axis=1)
        dataset['dependency'] = dataset.apply(lambda x: [], axis=1)
        dataset['sentence_attribute'] = dataset.apply(lambda x: [], axis=1)
    
    return train_data, test_data

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Extract data from MIMIC-IV transfers
    data = extract_data_from_transfers(args.mimic_path, args.window)
    
    # Add lab data if available
    data = add_lab_data(data, args.mimic_path)
    
    # Split data into train and test sets
    train_data, test_data = split_data(data, args.test_size, args.seed)
    
    # Process for concept extraction compatibility
    train_data, test_data = process_for_concept_extraction(train_data, test_data)
    
    # Save data
    print("Saving data...")
    train_data.to_csv(os.path.join(args.output_path, "train.csv"), index=False)
    test_data.to_csv(os.path.join(args.output_path, "test.csv"), index=False)
    
    print("Data preparation complete!")
    print(f"Data saved to {args.output_path}")
    print("You can now run the model using:")
    print(f"python run.py --dataset_folder {args.output_path} --output_dir ./model_output --multi_hop")

if __name__ == "__main__":
    main()