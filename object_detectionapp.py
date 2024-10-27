import streamlit as st
import numpy as np
import cv2  # For drawing bounding boxes
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained object detection model
model = load_model('ultralytics-coco8.keras')  # Ensure the model is for object detection

# Define the classes
classes = ['class1', 'class2']  # Replace with your actual class names

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((128, 128))  # Resize to match model input shape
    img_array = image.img_to_array(img) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Function to draw bounding boxes
def draw_boxes(image, boxes, scores, class_ids):
    for i in range(len(boxes)):
        box = boxes[i]
        x1, y1, x2, y2 = box
        label = f"{classes[class_ids[i]]}: {scores[i]:.2f}"
        
        # Draw rectangle
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2)
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Streamlit app layout
st.title("Object Detection")
st.write("Upload an image of an object:")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Load and display the image
    img = image.load_img(uploaded_file, target_size=(128, 128))
    img_array = preprocess_image(img)
    
    # Make predictions
    predictions = model.predict(img_array)  # This should return boxes, scores, and class_ids

    # Assuming predictions contain boxes, scores, and class IDs
    boxes, scores, class_ids = predictions  # Adjust based on your model's output format

    # Draw bounding boxes on the original image
    original_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    draw_boxes(original_image, boxes, scores, class_ids)
    
    # Convert back to RGB for Streamlit
    output_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Display the output image
    st.image(output_image, caption='Detected Image', use_column_width=True)