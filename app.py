import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

class_names = ['healthy', 'sick']

model = load_model('chicken_disease_model.h5')

st.title('Chicken Disease Detection')
st.write('Upload a chicken image to check its health status.')

uploaded_file = st.file_uploader('Upload Chicken Image', type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded image')

    if st.button('Predict'):
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        st.subheader(f'Prediction: {predicted_class}')
