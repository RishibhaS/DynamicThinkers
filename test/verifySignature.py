import tensorflow as tf
import numpy as np
import cv2
import os

IMG_HEIGHT = 128  # Update based on your model input
IMG_WIDTH = 128   # Update based on your model input
IMG_CHANNELS = 1  # For grayscale images; if RGB, set it to 3

def preprocess_signature_image(image_path):
    """
    Preprocess a signature image for the model.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.array: Preprocessed image ready for model input.
    """
    for img_name in os.listdir(image_path):
        img_path = os.path.join(image_path, img_name)
        try:
            # Load the image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Warning: Could not load {img_path}. Skipping.")
                continue

            # Resize the image
            img_resized = cv2.resize(img, (128,128))

            # Normalize to 0-1 range
            img_normalized = img_resized / 255.0;
            break
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    # Load the image as grayscale (use cv2.IMREAD_GRAYSCALE for single-channel images)
    #img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


    # Resize the image to match the model's expected input shape
   # img_resized = cv2.resize(img, (128, 128))

    # Normalize the pixel values (scale to [0, 1])
  #  img_resized = img_resized / 255.0  # Assuming model was trained with normalized data

    # Expand the dimensions to match the input format (e.g., (1, 128, 128, 1) for grayscale)
    img_resized = np.expand_dims(img_normalized, axis=-1)  # Add the channel dimension
    img_resized = np.expand_dims(img_normalized, axis=0)   # Add the batch dimension (e.g., (1, 128, 128, 1))

    return img_resized


# Function to predict if a signature is forged or genuine
def predict_signature(img_path):
    img = preprocess_signature_image(img_path)

    # Load the saved model
# Path to your saved model
    model_path = os.path.join('../models', 'signature_model.h5')  # Update with actual path

# Load the pre-trained model
    model = tf.keras.models.load_model(model_path)

# Display model summary (optional)
    model.summary()

    # Get model prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=0)[0]
    ##predicted_class = np.argmax(prediction, axis=1)[0] // forged signature


# 0 indicates genuine, 1 indicates forged
    if predicted_class == 0:
        return "Genuine"
    else:
        return "Forged"

def main():
    data_dir = '../data'
    raw_dir = os.path.join(data_dir, 'raw')
    image_dir = os.path.join(raw_dir, 'real')
    result = predict_signature(image_dir)
    print(f"The signature is: {result}")

# Test with a new signature
if __name__ == "__main__":
    main()