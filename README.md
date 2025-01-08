# RealTime-FaceMask-Detection
This repository contains a real-time face mask detection program built using Python, TensorFlow, Keras, and OpenCV. 
The project leverages a pre-trained deep learning model to detect whether a person is wearing a face mask or not through a live webcam feed.

-------

## **Project Structure**

- **`FaceMask_Detection (Updated).ipynb`**:  
  A Jupyter notebook containing the main program for real-time face mask detection. It handles loading the trained model, processing webcam input, and performing predictions.
  
- **`mask_detector2.keras`**:  
  The trained Keras model used for detecting face masks.

---

## **Requirements**

To run this project, ensure you have the following dependencies installed:

- Python 3.7+
- Jupyter Notebook
- TensorFlow 2.x
- Keras
- OpenCV
- NumPy
- Matplotlib
- imutils
- scikit-learn

Install the required Python packages using:

```bash
pip install tensorflow keras opencv-python-headless numpy matplotlib imutils scikit-learn
```

---

## Model Overview
The pre-trained model (mask_detector2.keras) was trained on a dataset containing images of people with and without face masks. The model was fine-tuned using MobileNetV2 as the base architecture, ensuring high accuracy and fast inference on real-time video streams.

## Machine Learning Models
**TensorFlow**:
TensorFlow is an open-source machine learning framework developed by Google. It allows developers to build and train deep learning models with ease, offering both flexibility and performance. In my project, TensorFlow provided the underlying tools for constructing and optimizing the neural network.

**Keras**:
Keras is a high-level API built on top of TensorFlow. It simplifies the process of building neural networks by providing an easy-to-use interface. In my project, I used Keras to define the model architecture, including the MobileNetV2 base model and custom layers.

**MobileNetV2**:
MobileNetV2 is a convolutional neural network architecture designed for mobile and embedded devices. It is efficient because it uses depthwise separable convolutions, significantly reducing computation while maintaining accuracy. Since real-time detection requires fast inference, MobileNetV2 was a great choice for my project.

### Key Features

**Real-time detection**: The program uses OpenCV to process frames from the webcam in real-time.

**High accuracy**: The MobileNetV2-based model achieves high precision and recall for mask detection.

**Data augmentation**: The model was trained with augmented data to improve its robustness to different lighting conditions and orientations.

---

## Future Improvements
**Improve robustness**: Add more diverse data for better generalization to unseen environments.

**Multi-face detection**: Enhance the model's capability to handle multiple faces in a single frame more effectively.

**Deploy to the web**: Convert the notebook into a standalone web application using Flask or Streamlit.

---

## Acknowledgments
- TensorFlow
- OpenCV
- Keras
