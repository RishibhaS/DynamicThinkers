import os

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from src.pretraining import update_train_folder, update_valid_folder
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.models import Sequential


def build_model(input_shape=(128, 128, 1)):
    """
    Builds a Convolutional Neural Network for binary classification.

    Args:
        input_shape (tuple): Shape of the input image (height, width, channels).

    Returns:
        model: A compiled Keras model.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')  # Binary classification (real vs forged)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def train_model(raw_dir,data_dir, batch_size=32, epochs=10, input_shape=(128, 128, 1)):
    """
    Train a CNN model using data from a directory.

    Args:
        data_dir (str): Path to the directory containing processed images.
        batch_size (int): Number of images per training batch.
        epochs (int): Number of training epochs.
        input_shape (tuple): Shape of the input image (height, width, channels).

    Returns:
        model: A trained Keras model.
    """
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')

    # Update Training
    update_train_folder(data_dir,train_dir)
    update_valid_folder(raw_dir,valid_dir)

    # Data augmentation for training
    train_datagen = ImageDataGenerator(rescale=1./255)
    valid_datagen = ImageDataGenerator(rescale=1./255)

    # Data generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='binary'
    )
    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='binary'
    )

    # Build and train the model
    model = build_model(input_shape)
    #print logs
    print("Classes in train_generator:", train_generator.class_indices)
    print("Number of validation samples:", valid_generator.samples)
    print("Number of train generating samples:", train_generator.samples)

    for batch_data, batch_labels in train_generator:
        print("Batch data shape:", batch_data.shape)
        print("Batch labels shape:", batch_labels.shape)
        break  # Check the first batch only

    for val_data, val_labels in valid_generator:
        print("Validation data shape:", val_data.shape)
        print("Validation labels shape:", val_labels.shape)
        break

    print("Training directory:", train_dir)
    print("Validation directory:", valid_dir)

    print("Found train samples:", len(os.listdir(train_dir)))
    print("Found validation samples:", len(os.listdir(valid_dir)))

    for img, label in train_generator:
        print("Image batch shape:", img.shape)
        print("Label batch shape:", label.shape)
        break

    model.summary()

    try:
        model.fit(
            train_generator,
            epochs=epochs,
            validation_data=valid_generator
        )
    except Exception as e:
        print("Error during model training:", e)

    return model

def save_model(model, filepath):
    """
    Save the trained model to a file.

    Args:
        model: The trained Keras model.
        filepath (str): Path to save the model.
    """
    model.save(filepath)
    print(f"Model saved to {filepath}")

if __name__ == "__main__":
    # Example usage
    data_dir = "data/processed"
    model = train_model(data_dir, batch_size=32, epochs=10, input_shape=(128, 128, 1))
    save_model(model, "models/signature_model.h5")
