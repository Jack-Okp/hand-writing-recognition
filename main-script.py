import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Import custom modules
from mnist_preprocessing import preprocess_mnist_data, apply_data_augmentation
from handwriting_cnn import create_cnn_model, train_model, evaluate_model

def main():
    # Preprocessing
    X_train, y_train, X_test, y_test = preprocess_mnist_data()
    
    # Data Augmentation
    datagen = apply_data_augmentation(X_train)
    
    # Train Model
    model, history = train_model(X_train, y_train, X_test, y_test)
    
    # Evaluate Model
    evaluate_model(model, X_test, y_test)
    
    # Plot Training History
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    
    # Save Model for Web Deployment
    model.save('handwriting_model.h5')
    tf.keras.models.save_model(model, 'web_model')
    
    # Convert for Web Deployment
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)

if __name__ == "__main__":
    main()
