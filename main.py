from src.preevaluation import load_and_preprocess_images, save_as_npz
from src.preprocessing import preprocess_images
from src.training import train_model, save_model
from src.evaluation import evaluate_model
import os

def main():
    # Paths for data
    data_dir = 'data'
    processed_dir = os.path.join(data_dir, 'processed')
    raw_dir = os.path.join(data_dir, 'raw')
    model_dir = 'models'
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Preprocessing
    print("Preprocessing images...")
    preprocess_images(input_dir=raw_dir,
                      output_dir=processed_dir)



    # Training
    print("Training the model...")
    model = train_model(raw_dir, processed_dir)
    model_path = os.path.join(model_dir, 'signature_model.h5')
    save_model(model, model_path)


    image_data, labels = load_and_preprocess_images(os.path.join(processed_dir,'real'), os.path.join(processed_dir, 'forged'));
    output_file = os.path.join(processed_dir, 'signature_data.npz')



    # Save the processed data as an .npz file
    save_as_npz(image_data, labels, output_file)



    # Evaluation
    print("Evaluating the model...")
    evaluate_model(model_path, output_file )

if __name__ == "__main__":
    main()
