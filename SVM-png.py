import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm

def preprocess_png_data(png_dir):
    X = []
    y = []
    for root, dirs, files in os.walk(png_dir):
        for file in files:
            if file.endswith(".png"):
                # Load PNG image
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                # Resize image to a fixed size
                img = cv2.resize(img, (64, 64))  # Example resizing to 64x64 pixels
                
                # Perform any additional preprocessing steps here
                
                # Append preprocessed image to X
                X.append(img)
                
                # Extract label from filename or directory name if applicable
                # For example, if your filenames contain label information
                # you can extract it like this:
                # label = file.split("_")[0]
                # y.append(label)
                
    # Convert X to numpy array
    X = np.array(X)
    
    # Convert y to numpy array if applicable
    # y = np.array(y)
    
    return X, y

def train_classifier(X, y):
    # Step 4: Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    # Step 5: Define and train your classifier
    classifier = svm.SVC()
    classifier.fit(X_train, y_train)

    # Step 6: Evaluate the classifier
    accuracy = classifier.score(X_test, y_test)
    print("Accuracy:", accuracy)

if __name__ == '__main__':
    png_dir = "png"
    
    # Step 1: Preprocess PNG data
    X, y = preprocess_png_data(png_dir)
    
    # Step 2: Train classifier
    train_classifier(X, y)
