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
from get_data import get_paths
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV data
data = pd.read_csv("ADNI1_Baseline_3T_3_20_2024.csv")

# Extract relevant columns
labels = data['Group']

# Paths to NIfTI files using your function
data_dir = "ADNI_data"
image_paths = get_paths("ADNI1_Baseline_3T_3_20_2024.csv", data_dir)

# Load NIfTI images and resize them to a fixed shape
fixed_shape = (64, 64, 64)  # Example shape, adjust as needed
images = []
for path in image_paths:
    try:
        if os.path.exists(path):
            img = image.load_img(path)
            img_resized = resize(img.get_fdata(), fixed_shape, anti_aliasing=True)
            images.append(img_resized)
    except Exception as e:
        print(f"Error loading image {path}: {e}")

# Convert images to arrays
X = np.array(images)

images_per_group = labels.value_counts()
print("Number of images per group:")
print(images_per_group)

# Perform label encoding for the labels (convert labels to integers)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
print("Class 0:", label_encoder.classes_[0])
print("Class 1:", label_encoder.classes_[1])
print("Class 2:", label_encoder.classes_[2])

# Reshape data
X_flat = X.reshape(X.shape[0], -1)

# Oversampling using RandomOverSampler
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

# Train and evaluate Decision Tree
decision_tree = DecisionTreeClassifier()
dt_trainer = ModelTrainer(decision_tree, "Decision Tree")
dt_trainer.evaluate_model(X_resampled, y_resampled)
y_pred_dt = dt_trainer.train_and_predict(X_resampled, y_resampled)

# Train and evaluate Random Forest
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
rf_trainer = ModelTrainer(random_forest, "Random Forest")
rf_trainer.evaluate_model(X_resampled, y_resampled)
y_pred_rf = rf_trainer.train_and_predict(X_resampled, y_resampled)

# Train and evaluate SVM
svm = SVC(kernel='linear', random_state=42)
svm_trainer = ModelTrainer(svm, "Support Vector Machine")
svm_trainer.evaluate_model(X_resampled, y_resampled)
y_pred_svm = svm_trainer.train_and_predict(X_resampled, y_resampled)

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=['AD', 'CN', 'MCI'], yticklabels=['AD', 'CN', 'MCI'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.show()

# Plot confusion matrices for each model
plot_confusion_matrix(y_resampled, y_pred_dt, "Decision Tree")
plot_confusion_matrix(y_resampled, y_pred_rf, "Random Forest")
plot_confusion_matrix(y_resampled, y_pred_svm, "Support Vector Machine")
