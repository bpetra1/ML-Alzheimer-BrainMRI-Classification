import os
import pandas as pd
from datetime import datetime

def get_paths_and_counts(csv_path, source_path):
    csv = pd.read_csv(csv_path)
    filepaths = []
    counts = {'AD': {'M': 0, 'F': 0}, 'MCI': {'M': 0, 'F': 0}, 'CN': {'M': 0, 'F': 0}}
    used_subjects = set()
    
    for key, value in csv.iterrows():
        # Filter descriptions that end with "Scaled"
        if value['Description'].endswith("Scaled"):
            modality_description = value['Description'].replace(';', '_').replace(' ', '_')
            acq_date = datetime.strptime(value['Acq Date'], '%m/%d/%Y').strftime('%Y-%m-%d')
    
            subject_path = os.path.join(source_path, value['Subject'], modality_description, acq_date, value['Image Data ID'])
            
            # Print constructed path for debugging
            #print(f"Constructed subject path: {subject_path}")
            
            # Check if the path exists before attempting to scan it
            if os.path.exists(subject_path):
                #print(f"Subject path exists: {subject_path}")
                with os.scandir(subject_path) as entries:
                    # Get the name of the first file in the directory
                    file_name = next(entries).name
                    # Construct the full path to the file
                    fmri_path = os.path.join(subject_path, file_name)
                    # Add the file path to the list
                    filepaths.append(fmri_path)
                    
                    # Update counts and used subjects set
                    group = value['Group']
                    sex = value['Sex']
                    if (value['Subject'], group, sex) not in used_subjects:
                        counts[group][sex] += 1
                        used_subjects.add((value['Subject'], group, sex))
            else:
                print(f"Subject path does not exist: {subject_path}")
                
    return filepaths, counts

# Example usage
csv_path = "./ADNI1_Baseline_3T_3_20_2024.csv"
source_path = "./ADNI_data"
file_paths, counts = get_paths_and_counts(csv_path, source_path)

# Print results
#print(f"File paths: {file_paths}")
#print(f"Counts: {counts}")
