import os
import numpy as np
import pandas as pd
from nilearn import image
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from skimage.transform import resize
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def get_paths(csv_path, source_path):
    csv = pd.read_csv(csv_path)
    filepaths = []
    for key, value in csv.iterrows():
        if value['Description'].endswith("Scaled"):
            modality_description = value['Description'].replace(';', '_').replace(' ', '_')
            acq_date = datetime.strptime(value['Acq Date'], '%m/%d/%Y').strftime('%Y-%m-%d')
            subject_path = os.path.join(source_path, value['Subject'], modality_description, acq_date, value['Image Data ID'])
            if os.path.exists(subject_path):
                with os.scandir(subject_path) as entries:
                    file_name = next(entries).name
                    smri_path = os.path.join(subject_path, file_name)
                    filepaths.append(smri_path)
    return filepaths

csv_path = "./ADNI1_Baseline_3T_3_20_2024.csv"
source_path = "./ADNI_data"
image_paths = get_paths(csv_path, source_path)

fixed_shape = (64, 64, 64)
images = []
for path in image_paths:
    try:
        if os.path.exists(path):
            img = image.load_img(path)
            img_resized = resize(img.get_fdata(), fixed_shape, anti_aliasing=True)
            images.append(img_resized)
    except Exception as e:
        print(f"Error loading image {path}: {e}")

X = np.array(images)

data = pd.read_csv(csv_path)
labels = data['Group'][data['Description'].str.endswith("Scaled")]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
print("Class 0:", label_encoder.classes_[0])
print("Class 1:", label_encoder.classes_[1])
print("Class 2:", label_encoder.classes_[2])

X_flat = X.reshape(X.shape[0], -1)

oversampler = RandomOverSampler()
X_resampled, y_resampled = oversampler.fit_resample(X_flat, y)

class ModelTrainer:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.y_pred = None

    def evaluate_model(self, X, y):
        print(f"\nEvaluating {self.model_name}...")
        scores = cross_val_score(self.model, X, y, cv=5)
        print(f"{self.model_name} cross-validation scores:", scores)
        print(f"Mean {self.model_name} cross-validation score:", scores.mean())
        return scores

    def train_and_predict(self, X, y):
        self.model.fit(X, y)
        self.y_pred = cross_val_predict(self.model, X, y, cv=5)
        print(f"\n{self.model_name} classification report:")
        print(classification_report(y, self.y_pred, zero_division=1))
        print(f"\n{self.model_name} confusion matrix: ")
        print(confusion_matrix(y, self.y_pred))
        return self.y_pred

decision_tree = DecisionTreeClassifier()
dt_trainer = ModelTrainer(decision_tree, "Decision Tree")
dt_trainer.evaluate_model(X_resampled, y_resampled)
y_pred_dt = dt_trainer.train_and_predict(X_resampled, y_resampled)

random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
rf_trainer = ModelTrainer(random_forest, "Random Forest")
rf_trainer.evaluate_model(X_resampled, y_resampled)
y_pred_rf = rf_trainer.train_and_predict(X_resampled, y_resampled)

svm = SVC(kernel='linear', random_state=42)
svm_trainer = ModelTrainer(svm, "Support Vector Machine")
svm_trainer.evaluate_model(X_resampled, y_resampled)
y_pred_svm = svm_trainer.train_and_predict(X_resampled, y_resampled)

def plot_confusion_matrix(y_true, y_pred, model_name):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.show()

plot_confusion_matrix(y_resampled, y_pred_dt, "Decision Tree")
plot_confusion_matrix(y_resampled, y_pred_rf, "Random Forest")
plot_confusion_matrix(y_resampled, y_pred_svm, "Support Vector Machine")
