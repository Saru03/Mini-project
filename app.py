import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import google.generativeai as genai

model_path = 'Mini_project_final2.h5'
model = load_model('Mini_project_final2.h5', compile=False)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

IMAGE_SIZE = 224
class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

def load_and_preprocess_image(image, image_size):
    img = Image.open(image)
    img = img.resize((image_size, image_size))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image, model, class_names, image_size):
    img_array = load_and_preprocess_image(image, image_size)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

st.title('Eye Disease Classification using Deep Learning')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=False, width=300)  # Adjust width here
    predicted_class, confidence = predict(uploaded_file, model, class_names, IMAGE_SIZE)
    st.write(f"Prediction: Class: {predicted_class}, Confidence: {confidence:.2f}%")

genai.configure(api_key="AIzaSyAD8ISLgPIV6TXKE4sLmYioO4A1NYRqeTw")

def get_gemini_response(question):
    response = genai.generate_text(prompt=question)
    return response.result

st.header("Eye Disease Chatbot")

input_text = st.text_input("Ask a question:", key="input_text")

submit = st.button("Ask the question")

if submit:
    response = get_gemini_response(input_text)
    st.subheader("The Response is")
    st.write(response)
