import os
from PIL import Image

def update_train_folder(raw_images_dir, train_dir):
    """
    Updates the train directory with processed images.

    Args:
        raw_images_dir (str): Directory containing raw images.
        train_dir (str): Path to the train directory (e.g., 'data/processed/train').
    """
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    real_dir = os.path.join(train_dir, 'real')
    forged_dir = os.path.join(train_dir, 'forged')

    # Create subdirectories for 'real' and 'forged' if they don't exist
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(forged_dir, exist_ok=True)

    # Process and move images to respective directories
    for category in ['real', 'forged']:
        category_dir = os.path.join(raw_images_dir, category)
        destination_dir = real_dir if category == 'real' else forged_dir

        # Process each image in the category
        for image_name in os.listdir(category_dir):
            image_path = os.path.join(category_dir, image_name)
            try:
                # Open image, process it (resize, convert to grayscale, etc.), and save
                img = Image.open(image_path)
                img = img.convert('L')  # Convert to grayscale if needed
                img = img.resize((128, 128))  # Resize image

                # Save processed image to train folder
                processed_image_path = os.path.join(destination_dir, image_name)
                img.save(processed_image_path)

            except Exception as e:
                print(f"Error processing {image_name}: {e}")


def update_valid_folder(raw_images_dir, valid_dir):
    """
    Updates the validation directory with processed images.

    Args:
        raw_images_dir (str): Directory containing raw validation images.
        valid_dir (str): Path to the validation directory (e.g., 'data/processed/valid').
    """
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)

    real_dir = os.path.join(valid_dir, 'real')
    forged_dir = os.path.join(valid_dir, 'forged')

    # Create subdirectories for 'real' and 'forged' if they don't exist
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(forged_dir, exist_ok=True)

    # Process and move images to respective directories
    for category in ['real', 'forged']:
        category_dir = os.path.join(raw_images_dir, category)
        destination_dir = real_dir if category == 'real' else forged_dir

        # Process each image in the category
        for image_name in os.listdir(category_dir):
            image_path = os.path.join(category_dir, image_name)
            try:
                # Open image, process it (resize, convert to grayscale, etc.), and save
                img = Image.open(image_path)
                img = img.convert('L')  # Convert to grayscale if needed
                img = img.resize((128, 128))  # Resize image

                # Save processed image to validation folder
                processed_image_path = os.path.join(destination_dir, image_name)
                img.save(processed_image_path)

            except Exception as e:
                print(f"Error processing {image_name}: {e}")



# Example usage
raw_images_dir = 'data/raw'  # Directory where raw images are stored
train_dir = 'data/processed/train'  # Directory where processed images will be saved
update_train_folder(raw_images_dir, train_dir)
