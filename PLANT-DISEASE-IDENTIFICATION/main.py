import streamlit as st
import os
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown

# -------------------------------
# Sidebar menu
# -------------------------------
st.sidebar.title("CropAssist Menu")
app_mode = st.sidebar.selectbox("Choose Feature", ["Crop Recommendation", "Fertilizer Recommendation", "Plant Disease Identification"])

# -------------------------------
# --------- CROP RECOMMENDATION ---------
# -------------------------------
if app_mode == "Crop Recommendation":
    st.title("Crop Recommendation")
    nitrogen = st.number_input("Nitrogen content in soil (N)")
    phosphorus = st.number_input("Phosphorus content in soil (P)")
    potassium = st.number_input("Potassium content in soil (K)")
    temperature = st.number_input("Temperature (°C)")
    humidity = st.number_input("Humidity (%)")
    ph = st.number_input("Soil pH")
    rainfall = st.number_input("Rainfall (mm)")
    
    if st.button("Predict Crop"):
        # Add your trained crop recommendation model prediction here
        # For example:
        st.success("Predicted Crop: Wheat")  # placeholder

# -------------------------------
# --------- FERTILIZER RECOMMENDATION ---------
# -------------------------------
elif app_mode == "Fertilizer Recommendation":
    st.title("Fertilizer Recommendation")
    nitrogen = st.number_input("Nitrogen content in soil (N)")
    phosphorus = st.number_input("Phosphorus content in soil (P)")
    potassium = st.number_input("Potassium content in soil (K)")
    temperature = st.number_input("Temperature (°C)")
    humidity = st.number_input("Humidity (%)")
    ph = st.number_input("Soil pH")
    rainfall = st.number_input("Rainfall (mm)")
    
    if st.button("Recommend Fertilizer"):
        # Add your trained fertilizer model prediction here
        st.success("Recommended Fertilizer: NPK")  # placeholder

# -------------------------------
# --------- PLANT DISEASE IDENTIFICATION ---------
# -------------------------------
elif app_mode == "Plant Disease Identification":
    st.title("Plant Disease Identification")
    
    MODEL_FOLDER = "trained_plant_disease_model"
    MODEL_FILENAME = "model.weights.h5"
    MODEL_PATH = os.path.join(MODEL_FOLDER, MODEL_FILENAME)
    GDRIVE_FILE_ID = "16EUJfdr8yMbjRlyR_TKa7lJGNcki9pYC"
    GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

    # Download model if missing
    if not os.path.exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)
    if not os.path.exists(MODEL_PATH):
        st.warning("Downloading model from Google Drive...")
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
        st.success("Model downloaded!")

    # Load model
    @st.cache_data
    def load_model_safe():
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except ValueError:
            model = build_model_architecture()
            model.load_weights(MODEL_PATH)
            return model

    # Build CNN architecture
    def build_model_architecture():
        cnn = tf.keras.models.Sequential()
        cnn.add(tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(128,128,3)))
        cnn.add(tf.keras.layers.Conv2D(32, 3, activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D(2,2))
        cnn.add(tf.keras.layers.Conv2D(64,3,padding='same',activation='relu'))
        cnn.add(tf.keras.layers.Conv2D(64,3,activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D(2,2))
        cnn.add(tf.keras.layers.Conv2D(128,3,padding='same',activation='relu'))
        cnn.add(tf.keras.layers.Conv2D(128,3,activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D(2,2))
        cnn.add(tf.keras.layers.Conv2D(256,3,padding='same',activation='relu'))
        cnn.add(tf.keras.layers.Conv2D(256,3,activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D(2,2))
        cnn.add(tf.keras.layers.Conv2D(512,3,padding='same',activation='relu'))
        cnn.add(tf.keras.layers.Conv2D(512,3,activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D(2,2))
        cnn.add(tf.keras.layers.Dropout(0.25))
        cnn.add(tf.keras.layers.Flatten())
        cnn.add(tf.keras.layers.Dense(1500,activation='relu'))
        cnn.add(tf.keras.layers.Dropout(0.4))
        cnn.add(tf.keras.layers.Dense(38,activation='softmax'))
        cnn.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        return cnn

    model = load_model_safe()
    st.success("Model loaded successfully!")

    uploaded_file = st.file_uploader("Upload a plant leaf image", type=["png","jpg","jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='Uploaded Image', use_column_width=True)
        img = img.resize((128,128))
        img_array = np.array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        class_index = np.argmax(pred, axis=1)[0]
        CLASS_NAMES = [...]  # Use your 38 class names here
        st.write(f"Predicted Disease: **{CLASS_NAMES[class_index]}**")
