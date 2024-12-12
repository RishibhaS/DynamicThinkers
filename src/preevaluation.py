import os
import numpy as np
from PIL import Image

def load_and_preprocess_images(real_dir, forged_dir, target_size=(128, 128)):
    """
    Loads and preprocesses images from two directories (real and forged).

    Args:
        real_dir (str): Directory containing real (genuine) signature images.
        forged_dir (str): Directory containing forged signature images.
        target_size (tuple): Desired size for the images, default is (128, 128).

    Returns:
        np.ndarray: Preprocessed image data (both real and forged).
        np.ndarray: Corresponding labels for real and forged signatures.
    """
    image_data = []
    labels = []

    # Load real images (label = 0)
    for filename in os.listdir(real_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(real_dir, filename)
            img = Image.open(img_path).resize(target_size).convert("L")
            img_array = np.array(img) / 255.0  # Normalize
            image_data.append(img_array)
            labels.append(0)  # Label 0 for real signatures

    # Load forged images (label = 1)
    for filename in os.listdir(forged_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(forged_dir, filename)
            img = Image.open(img_path).resize(target_size).convert("L")
            img_array = np.array(img) / 255.0  # Normalize
            image_data.append(img_array)
            labels.append(1)  # Label 1 for forged signatures

    # Convert lists to numpy arrays
    image_data = np.array(image_data)
    labels = np.array(labels)

    return image_data, labels

def save_as_npz(image_data, labels, output_path):
    """
    Saves image data and labels as an .npz file.

    Args:
        image_data (np.ndarray): The image data (numpy array).
        labels (np.ndarray): The labels (numpy array).
        output_path (str): Path to save the .npz file.
    """
    np.savez(output_path, images=image_data, labels=labels)
    print(f"Data saved as {output_path}")
    # Load the .npz file
    data = np.load(output_file, allow_pickle=True)  # use allow_pickle if data contains objects

    # Print the keys and check their contents
    for key in data.files:
        print(key, data[key])

# Example usage:
real_dir = 'data/processed/real'  # Directory for real signatures
forged_dir = 'data/processed/forged'  # Directory for forged signatures
output_file = 'data/processed/signature_data.npz'

# Load and preprocess both real and forged images
image_data, labels = load_and_preprocess_images(real_dir, forged_dir)

# Save the processed data as an .npz file
save_as_npz(image_data, labels, output_file)
