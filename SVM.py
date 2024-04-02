import os
import numpy as np
import pandas as pd
from nilearn import plotting, image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from skimage.transform import resize
from get_data import get_paths

# Load the CSV data
data = pd.read_csv("ADNI1_Baseline_3T_3_20_2024.csv")

# Extract relevant columns
subjects = data['Subject']
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

# Perform label encoding for the labels (convert labels to integers)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
print("Class 0:", label_encoder.classes_[0])
print("Class 1:", label_encoder.classes_[1])
print("Class 2:", label_encoder.classes_[2])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Reshape data for SVM
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Initialize and train SVM classifier
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Predict labels for test data
y_pred = svm.predict(X_test)

# Print classification report and confusion matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Visualize some example images
#plotting.plot_img(images[0])
#plotting.show()
