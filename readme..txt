

 Handwriting Recognition Project

 Overview
A machine learning project that develops a Convolutional Neural Network (CNN) for recognizing handwritten digits and characters using the MNIST dataset.

 Project Structure
- `mnist_preprocessing.py`: Data preprocessing and augmentation
- `handwriting_cnn.py`: CNN model architecture and training
- `main.py`: Project execution script
- `index.html`: Web application for character prediction

Technical Specifications
- **Model**: Convolutional Neural Network
- **Framework**: TensorFlow/Keras
- **Input**: 28x28 pixel grayscale images
- **Output**: Classification of digits (0-9)

Features
- Data augmentation techniques
- Dynamic learning rate adjustment
- Web-based prediction interface
- Client-side inference with TensorFlow.js

Installation
1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

Usage
1. Train the model:
   ```
   python main.py
   ```
2. Open `index.html` in a web browser for predictions

Dependencies
- TensorFlow
- NumPy
- Scikit-learn
- Matplotlib

Performance Metrics
- Training accuracy visualization
- Model evaluation on test dataset

 Future Enhancements
- Extend to letter recognition
- Implement advanced data augmentation
- Add confidence score display

