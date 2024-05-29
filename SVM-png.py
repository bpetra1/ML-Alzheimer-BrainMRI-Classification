import os
import numpy as np
import pandas as pd
from nilearn import image
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from skimage.transform import resize
from get_data import get_paths

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

# Reshape data for SVM
X_flat = X.reshape(X.shape[0], -1)

# Initialize SVM classifier
svm = SVC(kernel='linear', C=1.0)

# Initialize StratifiedKFold object
skf = StratifiedKFold(n_splits=3)

# Perform stratified k-fold cross-validation
scores = cross_val_score(svm, X_flat, y, cv=skf)

print("\nCross-validation scores:", scores)
print("Mean cross-validation score:", scores.mean())

# Train SVM classifier on the entire dataset
svm.fit(X_flat, y)

y_pred = cross_val_predict(svm, X_flat, y, cv=skf)

# Print classification report and confusion matrix
print(classification_report(y, y_pred, zero_division=1))
print("\nConfusion matrix: ")
print(confusion_matrix(y, y_pred))
