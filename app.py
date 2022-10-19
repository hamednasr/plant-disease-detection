import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('model.h5')
st.title('Tomato Plant Disease Detection')
st.subheader('First, upload a leaf plant photo and then click on Detect Disease.')
st.write('There are five kinds of diseases: **Bacterial spot, Target Spot, Leaf Mold, Early blight, Late blight**;')
st.write('and, **healthy** plant.')
classes = ['Tomato_Bacterial_spot',
            'Tomato_Early_blight',
            'Tomato_Late_blight',
            'Tomato_Leaf_Mold',
            'Tomato__Target_Spot',
            'Tomato_healthy']

def predict_disease(image):    
    prob = model.predict(image)
    pred_prob = np.max(prob, axis=1)[0]
    prediction = classes[np.argmax(prob, axis=1)[0]]
    return pred_prob, prediction

def main():
    img_file_buffer = st.file_uploader('Upload a leaf image here:', type='jpg')
    if img_file_buffer:
        img = Image.open(img_file_buffer)
        st.image(img)

    button = st.button('Detect Disease')
    if button:

        img = tf.keras.layers.Resizing(256,256)(img)
        img_array = np.expand_dims(np.array(img), axis=0)
        pred_prob, prediction = predict_disease(img_array)
        if prediction == 'healthy':
            st.success(f'The plant is healthy! with probability of %{pred_prob*100:.2f}')
        elif prediction != 'healthy':
            st.success(f'The plant is not healthy, with probability of %{pred_prob*100:.2f}, the disease is {prediction} ')
        

if __name__ == '__main__':
    main()