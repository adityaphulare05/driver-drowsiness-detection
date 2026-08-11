import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os
from imutils import paths
import errno

# Parameters
IMG_SIZE = (24, 24)
BATCH_SIZE = 32
EPOCHS = 20

def load_eye_data(data_path):
    images = []
    labels = []
    
    # Loop through open and closed eye directories
    for label, eye_state in enumerate(["open", "closed"]):
        eye_path = os.path.join(data_path, eye_state)
        
        # Check if directory exists and has images
        if not os.path.exists(eye_path):
            print(f"Warning: Directory {eye_path} does not exist!")
            continue
            
        image_files = list(paths.list_images(eye_path))
        if len(image_files) == 0:
            print(f"Warning: No images found in {eye_path}!")
            continue
            
        for image_path in image_files:
            try:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Warning: Could not read image {image_path}")
                    continue
                    
                image = cv2.resize(image, IMG_SIZE)
                image = image.astype("float32") / 255.0
                image = np.expand_dims(image, axis=-1)  # Add channel dimension
                
                images.append(image)
                labels.append(label)
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
    
    if len(images) == 0:
        raise ValueError("No images were loaded! Please check your data directories.")
    
    return np.array(images), np.array(labels)

def save_model_with_permission_handling(model, filepath):
    """Save model with proper permission handling"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Try to save the model
        model.save(filepath)
        print(f"Model successfully saved to {filepath}")
        return True
        
    except PermissionError:
        print(f"Permission denied for {filepath}. Trying alternative locations...")
        
        # Try saving to D drive directly
        try:
            d_drive_path = os.path.join("D:\\", os.path.basename(filepath))
            model.save(d_drive_path)
            print(f"Model saved to D drive: {d_drive_path}")
            return True
        except Exception as e:
            print(f"Failed to save to D drive: {e}")
            
        # Try saving to user documents folder
        try:
            documents_path = os.path.join(os.path.expanduser("~"), "Documents", os.path.basename(filepath))
            model.save(documents_path)
            print(f"Model saved to alternative location: {documents_path}")
            return True
        except Exception as e:
            print(f"Failed to save to alternative location: {e}")
            
        return False
        
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

# Load data
print("Loading eye data...")
try:
    X, y = load_eye_data("data/eyes")
    print(f"Loaded {len(X)} images")
except Exception as e:
    print(f"Error loading data: {e}")
    print("Please make sure you have collected eye data first using collect_eye_data.py")
    exit(1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {len(X_train)} images, Test set: {len(X_test)} images")

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)

# Create model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification: open or closed
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
print("Training model...")
try:
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=max(1, len(X_train) // BATCH_SIZE),  # Ensure at least 1 step
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=1
    )
except Exception as e:
    print(f"Error during training: {e}")
    exit(1)

# Save model with permission handling - SPECIFIC PATH FOR D DRIVE
model_path = r"D:\driver_drowsiness\models\eye_state_model.h5"
print(f"Attempting to save model to: {model_path}")
success = save_model_with_permission_handling(model, model_path)

if not success:
    print("\n" + "="*50)
    print("WARNING: Could not save model to preferred location.")
    print("Trying to save with a different approach...")
    
    # Try using the newer .keras format
    try:
        keras_model_path = r"D:\driver_drowsiness\models\eye_state_model.keras"
        os.makedirs(os.path.dirname(keras_model_path), exist_ok=True)
        model.save(keras_model_path)
        print(f"Model saved as .keras format: {keras_model_path}")
        print("Please update main.py to load from this file instead:")
        print('eye_model = load_model(r"D:\\driver_drowsiness\\models\\eye_state_model.keras")')
    except Exception as e:
        print(f"Also failed to save as .keras format: {e}")
        
        # Try saving to current directory as last resort
        try:
            model.save("eye_state_model.h5")
            print("Model saved as eye_state_model.h5 in current directory")
        except Exception as e:
            print(f"Failed to save to current directory: {e}")
            print("Model could not be saved anywhere!")

# Evaluate model
print("\nEvaluating model...")
try:
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {accuracy*100:.2f}%")
    print(f"Test loss: {loss:.4f}")
except Exception as e:
    print(f"Error during evaluation: {e}")

print("\nTraining completed!")