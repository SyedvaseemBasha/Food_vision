import streamlit as st
import tensorflow as tf
import numpy as np
from utils import load_and_prep_image, plot_top_10_probs, load_image_from_url, load_image_from_base64, create_temp_folder, preprocess_and_predict
# from config import MODEL_PATH
import os
import re

# Set the page configuration to wide mode
st.set_page_config(layout="wide")

# Load the trained model
MODEL_PATH="https://github.com/SyedvaseemBasha/Food_vision/blob/main/efficient_netb0_fine_tuned.keras"
model = tf.keras.models.load_model(MODEL_PATH)

import streamlit as st
import tensorflow as tf
import requests
import os

# Define model URL (raw file link)
MODEL_URL = "https://github.com/SyedvaseemBasha/Food_vision/raw/main/efficient_netb0_fine_tuned.keras"
MODEL_PATH = MODEL_URL

# Download the model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model... Please wait.")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    st.success("Model downloaded successfully!")

# Load the model
st.info("Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")


class_names = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap',
                'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 
                'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake',
                'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame',
                'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon',
                'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
                'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole',
                'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
                'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos',
                'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho',
                'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake',
                'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese',
                'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki',
                'tiramisu', 'tuna_tartare', 'waffles']

st.title("Food Image Classification")
st.write("Upload an image of food, and the model will predict its category.")

# Create a two-column layout
cols = st.columns([1, 2])

create_temp_folder("temp")

with cols[0]:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"], accept_multiple_files=False)
    if st.button("Show Trained image Class Names"):
        st.write(class_names)

with cols[1]:
    try:
        if uploaded_file is not None:
            temp_file_path = os.path.join("temp", uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            img = load_and_prep_image(temp_file_path)
            pred_class, pred_prob = preprocess_and_predict(img, model, class_names)
            st.image(temp_file_path, caption=f'Predicted: {pred_class}', use_column_width=True)
            os.remove(temp_file_path)
    except Exception as e:
        st.error(f"Error: {e}")

if uploaded_file is not None:
    st.markdown("---")
    try:
        if uploaded_file is not None:
            fig = plot_top_10_probs(pred_prob, class_names)
            st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown(
    """
    <style>
    footer {visibility: hidden;}
    .footer {visibility: visible; position: relative; bottom: 10px; text-align: center;}
    </style>
    """,
    unsafe_allow_html=True
)
