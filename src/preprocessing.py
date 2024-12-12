import os
import cv2
import numpy as np

def preprocess_images(input_dir, output_dir, img_size=(128, 128)):
    """
    Preprocess images: Resize, convert to grayscale, and normalize.

    Args:
        input_dir (str): Directory containing raw images (organized by class if applicable).
        output_dir (str): Directory to save preprocessed images.
        img_size (tuple): Target size for resizing images (width, height).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        output_class_path = os.path.join(output_dir, class_name)
        os.makedirs(output_class_path, exist_ok=True)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                # Load the image
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    print(f"Warning: Could not load {img_path}. Skipping.")
                    continue

                # Resize the image
                img_resized = cv2.resize(img, img_size)

                # Normalize to 0-1 range
                img_normalized = img_resized / 255.0

                # Save the preprocessed image
                save_path = os.path.join(output_class_path, img_name)
                cv2.imwrite(save_path, (img_normalized * 255).astype(np.uint8))

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    print(f"Preprocessing complete. Preprocessed images saved in {output_dir}")
