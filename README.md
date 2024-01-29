# Face Mask Detection using Convolutional Neural Networks

## Overview
This project focuses on building a Convolutional Neural Network (CNN) to detect whether a person is wearing a face mask or not. The dataset used for training and testing is obtained from Kaggle, comprising images labeled with and without face masks.

## Tools Used
- Python
- Jupyter Notebook
- Google Colab
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- TensorFlow
- Keras

## Project Workflow

1. **Data Collection:**
   - The Kaggle API is utilized to download the face mask dataset (`omkargurav/face-mask-dataset`).
   - The dataset is then extracted and organized for further processing.

2. **Data Exploration and Preprocessing:**
   - Image files are inspected to understand the structure of the dataset.
   - Images with and without masks are visualized for initial insights.

3. **Data Processing:**
   - Images are resized to 128x128 pixels and converted to RGB format.
   - Image data and corresponding labels are prepared for model training.

4. **Model Development:**
   - A CNN architecture is constructed using TensorFlow and Keras.
   - The model consists of convolutional layers, max-pooling layers, and dense layers.
   - The network is compiled using the Adam optimizer and sparse categorical crossentropy loss.

5. **Model Training:**
   - The dataset is split into training and testing sets.
   - Images are scaled, and the CNN is trained on the training set with 50 epochs.

6. **Model Evaluation:**
   - The model is evaluated on the test set, and accuracy metrics are calculated.
   - Training and validation loss and accuracy are visualized using Matplotlib.

7. **Prediction:**
   - A predictive system is implemented to classify an input image as with or without a face mask.
   - Users can input the path of an image to get real-time predictions.

## How to Use
1. Clone the repository.
2. Open the Jupyter Notebook in Google Colab or any compatible environment.
3. Run the notebook cells in sequence.

## Results
- The model achieves high accuracy on the test set, demonstrating its effectiveness in face mask detection.
- Visualizations provide insights into the training and validation process.

Feel free to contribute and enhance the project for broader applications in real-world scenarios.
