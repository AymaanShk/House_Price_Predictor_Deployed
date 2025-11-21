import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# --- Configuration and Setup ---

# Set Streamlit page configuration
st.set_page_config(
    page_title="Bangalore House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Function to load the cleaned data, caching it for performance
@st.cache_data
def load_data():
    """Loads the cleaned data to extract location names and unique feature values."""
    # Note: Assumes Cleaned_data.csv is available in the current directory
    data_path = 'Cleaned_data.csv'
    if not os.path.exists(data_path):
        st.error(f"Error: The required data file '{data_path}' was not found.")
        st.stop()
    
    try:
        df = pd.read_csv(data_path)
        # Convert 'bhk' to int for cleaner display, handling potential NaNs if necessary (though data is 'cleaned')
        df['bhk'] = df['bhk'].astype(int)
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        st.stop()

# Function to load the model pipeline, caching the resource
@st.cache_resource
def load_model():
    """Loads the trained machine learning pipeline (Ridge Model)."""
    # Note: Assumes RidgeModel.pk1 is available in the current directory
    model_path = 'RidgeModel.pk1'
    if not os.path.exists(model_path):
        st.error(f"Error: The required model file '{model_path}' was not found.")
        st.stop()
        
    try:
        with open(model_path, "rb") as file:
            pipe = pickle.load(file)
        return pipe
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        st.stop()

# Load the resources globally for the app
df = load_data()
pipe = load_model()

# Prepare unique values for selectboxes
locations = sorted(df['location'].unique())
bhk_options = sorted(df['bhk'].unique())
bath_options = sorted(df['bath'].unique())


# --- Main Application Layout ---

st.markdown("""
    <style>
    .st-emotion-cache-18ni7ap { /* Class for the main title wrapper */
        text-align: center;
    }
    .st-emotion-cache-1e5xgrd { /* Class for subheader */
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        padding: 10px;
        font-size: 1.1em;
        font-weight: bold;
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    .main-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        background-color: #f0f2f6;
    }
    .prediction-result {
        text-align: center;
        margin-top: 2rem;
        padding: 20px;
        border-radius: 10px;
        background-color: #e6ffe6;
        border: 2px solid #4CAF50;
    }
    </style>
    <div class="main-container">
    """, unsafe_allow_html=True)

st.title('🏠 Bengaluru House Price Predictor')
st.markdown("### Estimate the value of your property based on key features.")


# --- Input Form ---

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    location = st.selectbox(
        '📍 Select Location',
        locations,
        index=locations.index('Whitefield') if 'Whitefield' in locations else 0
    )

with col2:
    # Use st.number_input for continuous variable (Total Sqft)
    total_sqft = st.number_input(
        '📏 Total Square Feet (Sqft)',
        min_value=500.0,
        max_value=15000.0,
        value=1500.0,
        step=10.0,
        format="%.1f"
    )

col3, col4 = st.columns(2)

with col3:
    bhk = st.selectbox(
        '🛏️ Number of BHK',
        bhk_options,
        index=bhk_options.index(3) if 3 in bhk_options else 0
    )

with col4:
    # Bathroom count is often integer, but data can have floats (e.g., 3.0), so we keep options as is.
    bath = st.selectbox(
        '🛁 Number of Bathrooms',
        bath_options,
        index=bath_options.index(2.0) if 2.0 in bath_options else 0
    )

# --- Prediction Logic ---

st.markdown("</div>", unsafe_allow_html=True) # Close main-container

st.markdown("<br>", unsafe_allow_html=True)
if st.button('💰 Estimate Price'):
    # Input validation
    if total_sqft <= 0:
        st.error("Please enter a valid positive value for Total Square Feet.")
    else:
        try:
            # 1. Prepare input DataFrame with correct column names and data types (matching training data)
            input_df = pd.DataFrame(
                [[location, float(total_sqft), float(bath), int(bhk)]],
                columns=['location', 'total_sqft', 'bath', 'bhk']
            )

            # 2. Make Prediction
            prediction_base = pipe.predict(input_df)[0]

            # 3. Apply Scaling from original Flask App logic (assuming model output was scaled by 1e5)
            # Flask app used: prediction = pipe.predict(input)[0] * 1e5
            prediction_in_inr = prediction_base * 100000

            # 4. Display Result in Crores for better readability (1 Crore = 10,000,000 INR)
            price_in_crores = prediction_in_inr / 10000000

            # Display the result in a styled container
            st.markdown(f"""
                <div class="prediction-result">
                    <p style='font-size: 1.2em; color: #333;'>The estimated price is:</p>
                    <h1 style='color: #4CAF50; margin-top: 0;'>₹ {price_in_crores:,.2f} Crore</h1>
                    <p style='font-size: 0.9em; color: #777;'>
                        (This estimation is based on the Ridge Regression model trained on cleaned Bengaluru data.)
                    </p>
                </div>
            """, unsafe_allow_html=True)

        except ValueError:
            st.error("Invalid input detected. Please ensure all fields are correctly filled with numbers.")
        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: {e}")

# Add a section for data insights
st.sidebar.header("App Details")
st.sidebar.markdown(
    """
    This application uses a pre-trained **Ridge Regression Model**
    to estimate house prices in Bengaluru.

    **Model Inputs:**
    - Location (Categorical)
    - Total Square Feet (Numeric)
    - Number of Bathrooms (Numeric)
    - Number of BHK (Numeric/Integer)

    **Price Unit:** The output price is displayed in **Crores** (1 Crore = 10 Million INR).
    """

)
