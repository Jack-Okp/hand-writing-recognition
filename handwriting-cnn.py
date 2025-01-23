import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

def create_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Create a Convolutional Neural Network for handwriting recognition
    
    Args:
    - input_shape: Shape of input images
    - num_classes: Number of output classes
    
    Returns:
    - Compiled Keras model
    """
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(X_train, y_train, X_test, y_test):
    """
    Train the CNN model with hyperparameter tuning
    
    Args:
    - X_train, y_train: Training data and labels
    - X_test, y_test: Testing data and labels
    
    Returns:
    - Trained model
    - Training history
    """
    model = create_cnn_model()
    
    # Learning rate reducer
    lr_reducer = ReduceLROnPlateau(
        factor=0.5, 
        patience=3, 
        min_lr=0.00001
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=64,
        callbacks=[lr_reducer]
    )
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    Args:
    - model: Trained Keras model
    - X_test, y_test: Test data and labels
    
    Returns:
    - Evaluation metrics
    """
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    return test_loss, test_accuracy
