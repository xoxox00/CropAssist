import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ---------- Helper Functions ----------
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

def crop_recommendation(n, p, k, temp, hum, ph, rainfall):
    # Replace this logic with your trained crop recommendation model
    if n > 50 and p > 50 and k > 50:
        return "Wheat"
    else:
        return "Rice"

def fertilizer_recommendation(n, p, k):
    if n < 50:
        n_fert = "Nitrogen fertilizer recommended"
    else:
        n_fert = "Nitrogen level sufficient"
    if p < 30:
        p_fert = "Phosphorus fertilizer recommended"
    else:
        p_fert = "Phosphorus level sufficient"
    if k < 40:
        k_fert = "Potassium fertilizer recommended"
    else:
        k_fert = "Potassium level sufficient"
    return n_fert, p_fert, k_fert

# ---------- Sidebar ----------
st.sidebar.title("FarmAssistX")
app_mode = st.sidebar.selectbox("Select Page", [
    "HOME",
    "CROP RECOMMENDATION",
    "PLANT DISEASE DETECTION",
    "FERTILIZER RECOMMENDATION"
])

# ---------- HOME PAGE ----------
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>Welcome to FarmAssistX</h1>", unsafe_allow_html=True)

    img_path = "PLANT-DISEASE-IDENTIFICATION/test/Diseases.png"  # ✅ Corrected path
    if os.path.exists(img_path):
        img = Image.open(img_path)
        st.image(img)
    else:
        st.warning("⚠️ Diseases.png not found in test folder. Please check path.")

# ---------- CROP RECOMMENDATION ----------
elif app_mode == "CROP RECOMMENDATION":
    st.header("CROP RECOMMENDATION")
    nitrogen = st.number_input("Nitrogen content in soil (N)")
    phosphorus = st.number_input("Phosphorus content in soil (P)")
    potassium = st.number_input("Potassium content in soil (K)")
    temperature = st.number_input("Temperature (°C)")
    humidity = st.number_input("Humidity (%)")
    ph = st.number_input("Soil pH")
    rainfall = st.number_input("Rainfall (mm)")

    if st.button("Predict Crop"):
        recommended_crop = crop_recommendation(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
        st.success(f"🌱 Recommended Crop: {recommended_crop}")

# ---------- PLANT DISEASE DETECTION ----------
elif app_mode == "PLANT DISEASE DETECTION":
    st.header("DISEASE RECOGNITION")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image") and test_image:
        st.image(test_image, use_column_width=True)
    if st.button("Predict") and test_image:
        st.snow()
        result_index = model_prediction(test_image)
        class_name = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
            'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
        st.success(f"🤖 Model is Predicting it's a **{class_name[result_index]}**")

# ---------- FERTILIZER RECOMMENDATION ----------
elif app_mode == "FERTILIZER RECOMMENDATION":
    st.header("FERTILIZER RECOMMENDATION")
    nitrogen = st.number_input("Nitrogen content in soil (N)", key='fert_n')
    phosphorus = st.number_input("Phosphorus content in soil (P)", key='fert_p')
    potassium = st.number_input("Potassium content in soil (K)", key='fert_k')

    if st.button("Recommend Fertilizer"):
        n_msg, p_msg, k_msg = fertilizer_recommendation(nitrogen, phosphorus, potassium)
        st.success("🌿 " + n_msg)
        st.success("🌿 " + p_msg)
        st.success("🌿 " + k_msg)
