import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from io import BytesIO
from PIL import Image
import base64
import requests
import os

# Function to download files from Google Drive
def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

    # Verify if the downloaded file is a valid pickle file
    if not is_valid_pickle_file(destination):
        os.remove(destination)
        raise ValueError(f"Downloaded file '{destination}' is not a valid pickle file.")

def is_valid_pickle_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            pickle.load(file)
        return True
    except Exception:
        return False

# Function to load image as base64
def load_image_as_base64(image_path):
    try:
        image = Image.open(image_path)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return ""

# Functions for text processing
def lowercase(text):
    return text.lower()

def removepunc(text):
    return ''.join([char for char in text if char.isalnum() or char.isspace()])

def remove_sw(text):
    stopwords = set(['the', 'and', 'is', 'in', 'to', 'with'])
    return ' '.join([word for word in text.split() if word not in stopwords])

def stem_text(text):
    # Implement stemming logic here
    return text  # Replace with actual stemming logic

# Set the background colors and text color
st.markdown(
    """
    <style>
    body {
        background-color: #d3d3d3; /* Light gray background */
        margin: 0; /* Remove default margin for body */
        padding: 0; /* Remove default padding for body */
    }
    .st-bw {
        background-color: #eeeeee; /* Light gray background for widgets */
    }
    .st-cq {
        background-color: #cccccc; /* Gray background for chat input */
        border-radius: 10px; /* Add rounded corners */
        padding: 8px 12px; /* Add padding for input text */
        color: black; /* Set text color */
    }
    .st-cx {
        background-color: #d3d3d3; /* Light gray background for chat messages */
    }
    .sidebar .block-container {
        background-color: #d3d3d3; /* Light gray background for sidebar */
        border-radius: 10px; /* Add rounded corners */
        padding: 10px; /* Add some padding for spacing */
    }
    .top-right-image-container {
        position: fixed;
        top: 30px;
        right: 0;
        padding: 20px;
        background-color: #d3d3d3; /* Light gray background for image container */
        border-radius: 0 0 0 10px; /* Add rounded corners to bottom left */
    }
    .custom-font {
        color: #888888; /* Set font color to gray */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the logo image
logo_base64 = load_image_as_base64("BookGenie.png")

if logo_base64:
    st.markdown(
        f"""
        <div style='display: flex; align-items: center; gap: 15px;' class='custom-font'>
            <img src='data:image/png;base64,{logo_base64}' width='50'>
            <h1 style='margin: 0;'>BookGenie</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

# Text input for book description
book_description = st.text_area("Masukkan deskripsi buku:")

if st.button("Prediksi"):
    if not book_description:
        st.warning("Mohon isi deskripsi buku terlebih dahulu.")
    else:
        st.info("Sedang melakukan prediksi...")

        # Google Drive file IDs
        model_file_id = 'your_model_file_id'
        vectorizer_file_id = 'your_vectorizer_file_id'

        # File paths
        model_path = 'svm_model.pkl'
        vectorizer_path = 'tfidf_vectorizer.pkl'

        # Download files from Google Drive if they don't exist
        if not os.path.exists(model_path):
            try:
                download_file_from_google_drive(model_file_id, model_path)
            except ValueError as e:
                st.error(f"Error downloading the model: {e}")
                st.stop()

        if not os.path.exists(vectorizer_path):
            try:
                download_file_from_google_drive(vectorizer_file_id, vectorizer_path)
            except ValueError as e:
                st.error(f"Error downloading the vectorizer: {e}")
                st.stop()

        # Load SVM model and vectorizer
        try:
            with open(model_path, 'rb') as file:
                loaded_model = pickle.load(file)
        except Exception as e:
            st.error(f"Error loading the model: {e}")
            st.stop()
        
        try:
            with open(vectorizer_path, 'rb') as file:
                tfidf = pickle.load(file)
        except Exception as e:
            st.error(f"Error loading the vectorizer: {e}")
            st.stop()

        # Preprocess book description
        book_description_processed = [stem_text(remove_sw(removepunc(lowercase(book_description))))]

        # Transform book description using the loaded TfidfVectorizer
        try:
            book_description_tfidf = tfidf.transform(book_description_processed).toarray()
        except Exception as e:
            st.error(f"Error transforming the book description: {e}")
            st.stop()

        # Predict book genre
        try:
            predictions = loaded_model.predict(book_description_tfidf)
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.stop()

        # Map prediction to genre
        genre_mapping = {
            0: "adventure",
            1: "crime",
            2: "fantasy",
            3: "learning",
            4: "romance",
            5: "thriller"
        }
        
        predicted_genre = genre_mapping[predictions[0]]

        # Display prediction result
        st.write("Hasil Prediksi:")
        st.title(predicted_genre)
        st.success("Prediksi selesai!")
