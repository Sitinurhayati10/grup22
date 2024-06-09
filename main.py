import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from io import BytesIO
from PIL import Image
import base64
import zipfile
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

# Function to load image as base64
def load_image_as_base64(image_path):
    image = Image.open(image_path)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

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

st.markdown(
    f"""
    <div style='display: flex; align-items: center; gap: 15px;' class='custom-font'>
        <img src='data:image/png;base64,{logo_base64}' width='50'>
        <h1 style='margin: 0;'>BookGenie</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Kolom input teks untuk deskripsi buku
book_description = st.text_area("Masukkan deskripsi buku:")

if st.button("Prediksi"):
    if not book_description:
        st.warning("Mohon isi deskripsi buku terlebih dahulu.")
    else:
        st.info("Sedang melakukan prediksi...")

        # Google Drive file IDs
        model_file_id = 'your_model_file_id'
        vectorizer_file_id = 'your_vectorizer_file_id'
        xtrain_file_id = 'your_xtrain_file_id'

        # File paths
        model_zip_path = 'svm_model.zip'
        vectorizer_zip_path = 'tfidf_vectorizer.zip'
        xtrain_path = 'X_train_tfidf.csv'

        # Download files from Google Drive if they don't exist
        if not os.path.exists(model_zip_path):
            download_file_from_google_drive(model_file_id, model_zip_path)
        if not os.path.exists(vectorizer_zip_path):
            download_file_from_google_drive(vectorizer_file_id, vectorizer_zip_path)
        if not os.path.exists(xtrain_path):
            download_file_from_google_drive(xtrain_file_id, xtrain_path)

        # Unzip model and vectorizer files
        with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
            zip_ref.extractall()
        with zipfile.ZipFile(vectorizer_zip_path, 'r') as zip_ref:
            zip_ref.extractall()

        # Load model SVM and vectorizer
        with open("svm_model.pkl", 'rb') as file:
            loaded_model = pickle.load(file)
        with open("tfidf_vectorizer.pkl", 'rb') as file:
            tfidf = pickle.load(file)

        # Preprocessing deskripsi buku
        book_description_processed = [stem_text(remove_sw(removepunc(lowercase(book_description))))]

        # Membaca data X_train
        X_train = pd.read_csv(xtrain_path)  # Ubah sesuai dengan lokasi yang benar

        # Menerapkan pemrosesan teks pada data X_train
        X_train['Combined_Text'] = X_train['Combined_Text'].apply(lowercase)
        X_train['Combined_Text'] = X_train['Combined_Text'].apply(removepunc)
        X_train['Combined_Text'] = X_train['Combined_Text'].apply(remove_sw)
        X_train['Combined_Text'] = X_train['Combined_Text'].apply(stem_text)

        # Membuat dan melatih tfidf vectorizer dari data X_train
       
