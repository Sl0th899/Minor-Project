import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import os

def train_custom_cnn(dataset_path="dataset"):
    # 1. Configuration
    IMG_SIZE = (128, 128)
    BATCH_SIZE = 32
    EPOCHS = 15 # You can increase this later if it's still learning

    print("Loading dataset...")
    # Load training data (80% of images)
    train_ds = keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    
    # Load validation data (20% of images) to check accuracy during training
    val_ds = keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    # Save the class names so the Streamlit app knows what order they are in
    class_names = train_ds.class_names
    print(f"Found classes: {class_names}")
    with open("class_names.json", "w") as f:
        json.dump(class_names, f)

    # Performance optimization for loading images
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 2. Build the CNN Architecture from Scratch
    print("Building the CNN...")
    
    # Data Augmentation helps prevent memorization by tweaking the images slightly
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    model = keras.Sequential([
        data_augmentation,
        # Normalize pixel values from 0-255 to 0-1
        layers.Rescaling(1./255),
        
        # Layer 1: Looks for basic edges and colors
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        # Layer 2: Looks for simple shapes
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        # Layer 3: Looks for complex textures (like cardboard ribs)
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        # Flatten the 2D filters into a 1D list and make a decision
        layers.Flatten(),
        layers.Dropout(0.5), # Drops 50% of connections randomly to prevent overfitting
        layers.Dense(128, activation='relu'),
        layers.Dense(len(class_names), activation='softmax') # Final prediction probabilities
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # 3. Train the Model
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # 4. Save the completed brain
    print("Saving model to 'custom_waste_model.keras'...")
    model.save("custom_waste_model.keras")
    print("✅ Training complete!")

if __name__ == "__main__":
    train_custom_cnn()