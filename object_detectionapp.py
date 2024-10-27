import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load the trained model
model = load_model('/content/ultralytics-coco8.keras')

# Define the classes
classes = ['class1', 'class2']  # Replace with your actual class names

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((128, 128))  # Resize image to match model input shape
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Streamlit app layout
st.title("Hand Gesture Classification")
st.write("Upload an image of a hand gesture:")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Load and display the image
    img = image.load_img(uploaded_file, target_size=(128, 128))
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img_array = preprocess_image(img)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = classes[np.argmax(predictions)]

    # Display the prediction
    st.write(f'Predicted class: {predicted_class}')