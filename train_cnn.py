import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json

print("RUNNING NEW MOBILENET MODEL")

def train_custom_cnn(dataset_path="dataset"):

    # -----------------------------
    # CONFIG
    # -----------------------------

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 50

    print("Loading dataset...")

    # -----------------------------
    # LOAD DATASET
    # -----------------------------

    train_ds = keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    val_ds = keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = train_ds.class_names

    print(f"\nClasses found: {class_names}")

    # -----------------------------
    # SAVE CLASS NAMES
    # -----------------------------

    with open("class_names.json", "w") as f:
        json.dump(class_names, f)

    # -----------------------------
    # PERFORMANCE OPTIMIZATION
    # -----------------------------

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # -----------------------------
    # DATA AUGMENTATION
    # -----------------------------

    data_augmentation = keras.Sequential([

        layers.RandomFlip("horizontal"),

        layers.RandomRotation(0.2),

        layers.RandomZoom(0.2),

        layers.RandomContrast(0.2),

        layers.RandomTranslation(0.1, 0.1),

    ])

    # -----------------------------
    # LOAD PRETRAINED MOBILENET
    # -----------------------------

    print("\nLoading MobileNetV2...")

    base_model = tf.keras.applications.MobileNetV2(

        input_shape=(224, 224, 3),

        include_top=False,

        weights="imagenet"

    )

    # Freeze pretrained layers
    base_model.trainable = False

    # -----------------------------
    # BUILD MODEL
    # -----------------------------

    print("\nBuilding model...")

    inputs = keras.Input(shape=(224, 224, 3))

    x = data_augmentation(inputs)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        len(class_names),
        activation="softmax"
    )(x)

    model = keras.Model(inputs, outputs)

    # -----------------------------
    # COMPILE
    # -----------------------------

    model.compile(

        optimizer=keras.optimizers.Adam(learning_rate=0.001),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]

    )

    # -----------------------------
    # MODEL SUMMARY
    # -----------------------------

    model.summary()

    # -----------------------------
    # CALLBACKS
    # -----------------------------

    early_stop = keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=5,

        restore_best_weights=True

    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.2,

        patience=2,

        verbose=1

    )

    # -----------------------------
    # TRAIN
    # -----------------------------

    print("\nTraining model...\n")

    history = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=EPOCHS,

        callbacks=[early_stop, reduce_lr]

    )

    # -----------------------------
    # FINE TUNING
    # -----------------------------

    print("\nStarting fine-tuning...")

    base_model.trainable = True

    # Freeze lower layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(

        optimizer=keras.optimizers.Adam(learning_rate=0.00001),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]

    )

    fine_tune_epochs = 10

    total_epochs = EPOCHS + fine_tune_epochs

    history_fine = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=total_epochs,

        initial_epoch=history.epoch[-1],

        callbacks=[early_stop, reduce_lr]

    )

    # -----------------------------
    # SAVE MODEL
    # -----------------------------

    print("\nSaving model...")

    model.save("custom_waste_model.keras")

    print("\nTraining complete!")

    # -----------------------------
    # FINAL METRICS
    # -----------------------------

    final_acc = max(history.history["val_accuracy"])

    print(f"\nBest Validation Accuracy: {final_acc * 100:.2f}%")

if __name__ == "__main__":
    train_custom_cnn()