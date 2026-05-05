import os
import pickle
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

def extract_features(image: Image.Image):
    image = image.resize((64, 64))
    
    # Texture (HOG)
    gray_image = np.array(image.convert("L"))
    hog_features = hog(gray_image, orientations=8, pixels_per_cell=(16, 16),
                       cells_per_block=(1, 1), feature_vector=True)
    
    # Color (HSV)
    hsv_image = image.convert("HSV")
    h, s, v = hsv_image.split()
    h_hist, _ = np.histogram(np.array(h).flatten(), bins=16, range=(0, 256))
    s_hist, _ = np.histogram(np.array(s).flatten(), bins=8, range=(0, 256))
    v_hist, _ = np.histogram(np.array(v).flatten(), bins=8, range=(0, 256))
    color_features = np.concatenate([h_hist, s_hist, v_hist]) / (64 * 64)

    return np.concatenate([hog_features, color_features])

def train_model(dataset_path="dataset"):
    X = []
    y = []
    classes = []
    
    print("Extracting features from images...")
    
    # Loop through folders in the dataset directory
    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        classes.append(class_name)
        
        for file_name in os.listdir(class_dir):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, file_name)
                try:
                    img = Image.open(img_path).convert("RGB")
                    features = extract_features(img)
                    X.append(features)
                    y.append(class_name)
                except Exception as e:
                    print(f"Error loading {file_name}: {e}")

    X = np.vstack(X)
    y = np.array(y)
    
    print(f"Extracted {len(X)} samples. Scaling data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Training Random Forest with 100 trees...")
    # Upgrade: n_estimators=100 instead of 10. class_weight="balanced" helps if you have uneven data.
    model = RandomForestClassifier(n_estimators=100, max_depth=None, class_weight="balanced", random_state=42)
    model.fit(X_scaled, y)
    
    print("Saving new model.pkl...")
    bundle = {
        "model": model,
        "scaler": scaler,
        "classes": model.classes_
    }
    
    with open("model.pkl", "wb") as f:
        pickle.dump(bundle, f)
        
    print("Training complete!")

if __name__ == "__main__":
    train_model()