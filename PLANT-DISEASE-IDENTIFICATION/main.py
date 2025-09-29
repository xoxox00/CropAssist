import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import streamlit as st
import gdown
from PIL import Image

# -------------------------------
# Paths and Google Drive File ID
# -------------------------------
MODEL_FOLDER = "trained_plant_disease_model"
MODEL_FILENAME = "model.weights.h5"
MODEL_PATH = os.path.join(MODEL_FOLDER, MODEL_FILENAME)

GDRIVE_FILE_ID = "16EUJfdr8yMbjRlyR_TKa7lJGNcki9pYC"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

# -------------------------------
# Download model if not exists
# -------------------------------
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

if not os.path.exists(MODEL_PATH):
    st.warning("Model file not found! Downloading from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    st.success("Model downloaded successfully!")

# -------------------------------
# Load the model
# -------------------------------
@st.cache_data(show_spinner=True)
def load_model_safe():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except ValueError:
        # If it’s just weights
        model = build_model_architecture()
        model.load_weights(MODEL_PATH)
        return model

# -------------------------------
# Build CNN architecture (matches training)
# -------------------------------
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
    cnn.add(tf.keras.layers.Dense(38,activation='softmax'))  # 38 classes

    cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return cnn

# -------------------------------
# Load model
# -------------------------------
st.title("Plant Disease Identification")
model = load_model_safe()
st.success("Model loaded successfully!")

# -------------------------------
# Upload image
# -------------------------------
uploaded_file = st.file_uploader("Upload a plant leaf image", type=["png","jpg","jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess image
    img = img.resize((128,128))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    pred = model.predict(img_array)
    class_index = np.argmax(pred, axis=1)[0]

    # Class names (must match training)
    CLASS_NAMES = ['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
                   'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
                   'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight',
                   'Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy',
                   'Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy',
                   'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight',
                   'Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

    st.write(f"Predicted Disease: **{CLASS_NAMES[class_index]}**")
