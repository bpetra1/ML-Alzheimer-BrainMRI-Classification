import os
import numpy as np
import pandas as pd
from nilearn import image
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from skimage.transform import resize
from get_data import get_paths
from imblearn.over_sampling import RandomOverSampler

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

    def evaluate_model(self, X, y):
        print(f"\nEvaluating {self.model_name}...")
        scores = cross_val_score(self.model, X, y, cv=5)
        print(f"{self.model_name} cross-validation scores:", scores)
        print(f"Mean {self.model_name} cross-validation score:", scores.mean())
        return scores

    def train_and_predict(self, X, y):
        self.model.fit(X, y)
        y_pred = cross_val_predict(self.model, X, y, cv=5)
        print(f"\n{self.model_name} classification report:")
        print(classification_report(y, y_pred, zero_division=1))
        print(f"\n{self.model_name} confusion matrix: ")
        print(confusion_matrix(y, y_pred))

# Train and evaluate Decision Tree
decision_tree = DecisionTreeClassifier()
dt_trainer = ModelTrainer(decision_tree, "Decision Tree")
dt_trainer.evaluate_model(X_resampled, y_resampled)
dt_trainer.train_and_predict(X_resampled, y_resampled)

# Train and evaluate Random Forest
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
rf_trainer = ModelTrainer(random_forest, "Random Forest")
rf_trainer.evaluate_model(X_resampled, y_resampled)
rf_trainer.train_and_predict(X_resampled, y_resampled)

# Train and evaluate K-Nearest Neighbors
#knn = KNeighborsClassifier(n_neighbors=5)
#knn_trainer = ModelTrainer(knn, "K-Nearest Neighbors")
#knn_trainer.evaluate_model(X_resampled, y_resampled)
#knn_trainer.train_and_predict(X_resampled, y_resampled)

# Train and evaluate SVM
svm = SVC(kernel='linear', random_state=42)
svm_trainer = ModelTrainer(svm, "Support Vector Machine")
svm_trainer.evaluate_model(X_resampled, y_resampled)
svm_trainer.train_and_predict(X_resampled, y_resampled)
