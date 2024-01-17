# Emotion Detection Using TensorFlow

## Overview

This project implements an Emotion Detection system using TensorFlow, a powerful open-source machine learning library. The system is designed to recognize facial expressions in images and classify them into seven different emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

## Project Structure

The project is organized into two main components:

1. **Model Creation (model_creation.ipynb):**
   - **Libraries Used:** Matplotlib, OpenCV, TensorFlow, IPython, etc.
   - **Exploring Dataset:** Randomly selects and displays images from the training dataset.
   - **Preparing Data for Training:** Uses the ImageDataGenerator to preprocess and augment training and validation data.
   - **Defining Model:** Defines a Convolutional Neural Network (CNN) for emotion detection.
   - **Initializing and Training the Model:** Initializes the model, trains it on the provided dataset, and saves the model weights.
   - **Model Evaluation:** Evaluates the model on the validation dataset and displays accuracy metrics.
   - **Plotting Loss and Accuracy:** Plots the training and validation loss and accuracy over epochs.
   - **Saving Model:** Saves the trained model architecture in JSON format.

2. **GUI Application (gui.py):**
   - **Libraries Used:** Tkinter, OpenCV, TensorFlow, PIL (Pillow), NumPy, etc.
   - **Facial Expression Model:** Loads the trained model from JSON and weights files.
   - **User Interface:** Utilizes Tkinter to create a simple GUI for uploading images and detecting emotions.
   - **Image Processing:** Uses OpenCV to detect faces in uploaded images and predict emotions using the trained model.
   - **Result Display:** Displays the detected emotion on the GUI.

## Usage

1. **Model Training:**
   - Open and run the `model_creation.ipynb` notebook to train the emotion detection model.
   - Adjust parameters such as image size, batch size, and epochs as needed.
   - The trained model weights will be saved in the `model_weights.h5` file.

2. **GUI Application:**
   - Run the `gui.py` script to launch the GUI for emotion detection.
   - Click the "Upload Image" button to select an image for emotion detection.
   - The detected emotion will be displayed on the GUI.

3. **Examples:**
   
![Screen Shot 2024-01-17 at 12 57 48 PM](https://github.com/VRAJ-07/Emotion-Detection-Using-Tensorflow/assets/86062890/7a41587e-a522-4e4a-adbf-9d1292aa4842)
![Screen Shot 2024-01-17 at 12 57 53 PM](https://github.com/VRAJ-07/Emotion-Detection-Using-Tensorflow/assets/86062890/b0e42d27-805c-4804-81ac-e92c2cde1b7e)
![Screen Shot 2024-01-17 at 12 57 41 PM](https://github.com/VRAJ-07/Emotion-Detection-Using-Tensorflow/assets/86062890/881022d1-ad42-479b-829d-92a96be72a04)
![Screen Shot 2024-01-17 at 12 57 59 PM](https://github.com/VRAJ-07/Emotion-Detection-Using-Tensorflow/assets/86062890/f5bcf551-9124-4d75-a077-114c8c4c7222)
![Screen Shot 2024-01-17 at 12 58 05 PM](https://github.com/VRAJ-07/Emotion-Detection-Using-Tensorflow/assets/86062890/910a6f32-5ab4-48ac-a916-75a91f725ced)


## Dependencies

Ensure you have the required dependencies installed. You can install them using the following commands:

```bash
pip install matplotlib opencv-python tensorflow pillow numpy
```

## Additional Notes

- The Haar Cascade Classifier file (`haarcascade_frontalface_default.xml`) is required for face detection and should be downloaded from [here](https://github.com/opencv/opencv/tree/master/data/haarcascades).
