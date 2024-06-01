import os
import pandas as pd
from datetime import datetime

def get_paths(csv_path, source_path):
    csv = pd.read_csv(csv_path)
    filepaths = []
    for key, value in csv.iterrows():
        modality_description = value['Description'].replace(';', '_').replace(' ', '_')
        acq_date = datetime.strptime(value['Acq Date'], '%m/%d/%Y').strftime('%Y-%m-%d')

        subject_path = os.path.join(source_path, value['Subject'], modality_description, acq_date, value['Image Data ID'])

        with os.scandir(subject_path) as entries:
            file_name = next(entries).name
        fmri_path = os.path.join(subject_path, file_name)
        filepaths.append(fmri_path)
    return filepaths

get_paths("ADNI1_Baseline_3T_3_20_2024.csv", "ADNI_data")
