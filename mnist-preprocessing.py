import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def preprocess_mnist_data():
    """
    Preprocess MNIST dataset for combined digit and letter recognition
    
    Returns:
    - X_train, X_test: preprocessed image data
    - y_train, y_test: one-hot encoded labels
    """
    # Load MNIST digit dataset
    (x_digits, y_digits), (x_digits_test, y_digits_test) = mnist.load_data()
    
    # Reshape and normalize
    x_digits = x_digits.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_digits_test = x_digits_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # Convert labels to one-hot encoding (0-9)
    y_digits = to_categorical(y_digits, 10)
    y_digits_test = to_categorical(y_digits_test, 10)
    
    # Note: For full implementation, you'd need to add letter datasets
    # This is a placeholder for extending beyond MNIST digits
    
    return x_digits, y_digits, x_digits_test, y_digits_test

def apply_data_augmentation(X_train):
    """
    Apply data augmentation techniques
    
    Args:
    - X_train: Training image data
    
    Returns:
    - Augmented training data
    """
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    datagen.fit(X_train)
    return datagen
