import os
import pandas as pd
from datetime import datetime

def get_paths(csv_path, source_path):
    csv = pd.read_csv(csv_path)
    #binary_class = []
    filepaths = []
    for key, value in csv.iterrows():
        #binary_class.append(value['Group'])  # Use 'Group' column as binary labels
        modality_description = value['Description'].replace(';', '_').replace(' ', '_')
        acq_date = datetime.strptime(value['Acq Date'], '%m/%d/%Y').strftime('%Y-%m-%d')

        subject_path = os.path.join(source_path, value['Subject'], modality_description, acq_date, value['Image Data ID'])

        # Assuming there's only one file in each folder, get the first file found
        with os.scandir(subject_path) as entries:
            file_name = next(entries).name
        fmri_path = os.path.join(subject_path, file_name)
        print(fmri_path)
        filepaths.append(fmri_path)
    return filepaths #, binary_class

get_paths("ADNI1_Baseline_3T_3_20_2024.csv", "ADNI_data")
