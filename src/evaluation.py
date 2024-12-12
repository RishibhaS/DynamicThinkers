import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

def load_data(file_path):
    """
    Load test data from a .npz file.

    Args:
        file_path (str): Path to the .npz file.

    Returns:
        Tuple: Test features and labels.
    """
    data = np.load(file_path)
    X_test = data['images']
    y_test = data['labels']
    #y_test = data['y_test']
    return X_test, y_test

def evaluate_model(model_path, data_path):
    """
    Evaluate the trained model on test data.

    Args:
        model_path (str): Path to the saved Keras model.
        data_path (str): Path to the test data (.npz file).
    """
    # Load the model
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")

    # Load test data
    X_test, y_test = load_data(data_path)

    # Reshape inputs if necessary
    X_1_test = X_test.reshape(-1, 128, 128, 1)  # Adjust if your input shape differs
   # X_2_test = X_2_test.reshape(-1, 128, 128, 1)

    # Evaluate the model
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Predict and generate metrics
    predictions = model.predict([X_1_test])
    y_pred = np.argmax(predictions, axis=1)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    model_path = "models/signature_model.h5"
    data_path = "data/processed/test.npz"

    evaluate_model(model_path, data_path)
