import streamlit as st
import cv2
from PIL import Image
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import random
import numpy as np
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tempfile


# Function to load and encode the image in base64
def get_base64_of_bin_file(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Provide the path to your local image
image_file_path = "music.jpg" 
base64_img = get_base64_of_bin_file(image_file_path)

# CSS for setting the background
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{base64_img}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
</style>
"""
st.sidebar.header("About Moodify")
st.sidebar.write("Moodify is an AI-powered music recommendation app that detects a user's mood through facial expressions and suggests songs accordingly. It captures an image using the webcam, analyzes the mood using a deep learning model, and fetches a matching song from Spotify. The app supports seven mood categories: Angry, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised.")
st.markdown(page_bg_img, unsafe_allow_html=True)

model = load_model("mood.h5")

# Define mood categories
mood_categories = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]

# Spotify API credentials (replace with your actual credentials)
SPOTIPY_CLIENT_ID = "Your Spotify client api key"
SPOTIPY_CLIENT_SECRET = "Your spotify secrete api key"

# Authenticate with Spotify
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID,client_secret=SPOTIPY_CLIENT_SECRET))

# Mood-based playlist mapping (Spotify playlist IDs)
MOOD_PLAYLISTS = {
    "Angry": "3JNWpteYvH3ynMcyPcvxfx",  # Replace with a valid playlist ID for Angry mood
    "Disgusted": "37i9dQZF1EIfwweG9kDJVK",  # Replace with a valid playlist ID for Disgusted mood
    "Fearful": "6azZ2wEJOeiQUKrEp1Hl2p",  # Replace with a valid playlist ID for Fearful mood
    "Happy": "37i9dQZF1DWTwbZHrJRIgD",  # Replace with a valid playlist ID for Happy mood
    "Neutral": "55PVuXcePN1SJUh8yczGuR",  # Replace with a valid playlist ID for Neutral mood
    "Sad": "2sOMIgioNPngXojcOuR4tn",  # Replace with a valid playlist ID for Sad mood
    "Surprised": "4E70J2NFke7FMwdWirzgKP"  # Replace with a valid playlist ID for Surprised mood
}


# Function to get song recommendation based on mood
def get_mood_song(mood):
    """Fetch a random song from the given mood's playlist."""
    if mood not in MOOD_PLAYLISTS:
        return "Mood not found!"
    
    playlist_id = MOOD_PLAYLISTS[mood]
    results = sp.playlist_tracks(playlist_id)
    
    tracks = results['items']
    if not tracks:
        return "No songs found for this mood."

    # Select a random song
    song = random.choice(tracks)['track']
    
    # Get album artwork (cover image)
    album_image = song['album']['images'][0]['url']

    song_details = f"🎵 {song['name']} by {', '.join([artist['name'] for artist in song['artists']])} \n🔗 {song['external_urls']['spotify']}"
    
    return song_details, album_image

# Streamlit Interface
st.title("Moodify - Capture Your Mood and Get Music Suggestions")

# Capture image using webcam
st.write("Click on the button below to capture a photo:")
image_file = st.camera_input("Take a Photo")

if image_file is not None:
    # Open the image
    img = Image.open(image_file)
    
    # Display the captured image
    st.image(img, caption="Captured Image", use_column_width=True)


    def preprocess_image(img):
        # Convert image to grayscale
        img = img.convert('L')  # 'L' mode ensures single-channel (grayscale)

        # Resize the image to (224, 224)
        img = img.resize((224, 224))

        # Convert image to array
        img_array = img_to_array(img)  # Shape: (224, 224, 1)
        img_array = img_array / 255.0  # Normalize to 0-1
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 1)

        return img_array

    process_img=preprocess_image(img)
    # For demo, let's assume you detect a random mood
    prediction = model.predict(process_img)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Get the predicted mood from the class index
    detected_mood = mood_categories[predicted_class]
    
    st.write(f"Detected Mood: {detected_mood}")

    # Fetch and display music recommendation
    with st.expander(label="See the Song Recommendation..",expanded=False):
        recommended_song, album_image = get_mood_song(detected_mood)
        st.image(album_image, use_column_width=True)
        st.write(recommended_song) 
else:
    st.write("Please take a photo to get started!")
