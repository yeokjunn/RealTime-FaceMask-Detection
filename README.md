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

Key Features
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
