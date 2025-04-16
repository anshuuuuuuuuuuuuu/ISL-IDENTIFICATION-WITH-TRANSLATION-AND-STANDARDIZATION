import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import time 
import pyttsx3
import requests
import streamlit as st
import pandas as pd
import sklearn from scikit-learn

# Set up the Gemini API key and endpoint from google gemini Website
api_key = "INSERT YOUR GEMINI API KEY HERE (OBTAIN IT FOR FREE FROM GOOGLE API DEVELOPER )"
endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"

# Initialize the text-to-speech engine 
engine = pyttsx3.init()
 
# Load the saved model 
model = load_model('action_recognition_model1.h5') 

# Define class names 
# Add more classes according to your custom dataset . This is an expandable project with multiple classes.
class_names = ['hello', 'this', 'prototype', 'demonstration']

# Initialize Mediapipe Hand model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Variables for predictions and recording state
last_prediction_time = 0
prediction_delay = 2  # seconds
predicted_label = ""
last_stored_prediction = None
predictions = []
recording = False  # Track if recording is active
no_hand_detected_time = None
no_hand_duration = 5  # Time duration for no hand detection

# Define the function to standardize the sentence using the Gemini API
def standardize_sentence(predictions, language):
    text = " ".join(predictions)
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"Correct the following sentence: {text}. Translate to {language} with correct grammar and add prepositions, etc. if reqired.return only one correct answer and not extra stuff like  explaination.for hello assume hello everyone and have only one hello in one sentence and generate only one sentence at time"
                    }
                ]
            }
        ]
    }
    response = requests.post(f'{endpoint}?key={api_key}', headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        corrected_text = result['candidates'][0]['content']['parts'][0]['text']
        return corrected_text
    else:
        return "Error in text generation."

# Streamlit UI elements
st.title("Sign Language Recognition App")
start_button = st.button("Start Recording")
stop_button = st.button("Stop Recording")
language = st.text_input("Preferred Language", "english")
frame_window = st.image([])  # Initialize a window for the video frames

# Start video capture
# Ensure that you have webcam enabled with proper permissions
cap = cv2.VideoCapture(0)

# Set up the hand model with Mediapipe for LandMarks
with mp_hands.Hands(min_detection_confidence=0.4, min_tracking_confidence=0.4) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame.")
            break
        
        # Handle recording state
        if start_button:
            recording = True
        elif stop_button:
            recording = False
        
        # If recording, process the frames from the webcam
        if recording:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            if results.multi_hand_landmarks:
                no_hand_detected_time = None
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    h, w, c = frame.shape
                    landmark_coords = [(int(point.x * w), int(point.y * h)) for point in hand_landmarks.landmark]

                    x_min = max(0, min([coord[0] for coord in landmark_coords]) - 20)
                    x_max = min(w, max([coord[0] for coord in landmark_coords]) + 20)
                    y_min = max(0, min([coord[1] for coord in landmark_coords]) - 20)
                    y_max = min(h, max([coord[1] for coord in landmark_coords]) + 20)

                    hand_region = frame[y_min:y_max, x_min:x_max]

                    current_time = time.time()
                    if current_time - last_prediction_time > prediction_delay:
                        input_image = cv2.resize(hand_region, (50, 50))
                        input_image = np.expand_dims(input_image, axis=0)
                        input_image = input_image.astype('float32') / 255.0

                        prediction = model.predict(input_image)
                        predicted_class = np.argmax(prediction, axis=1)
                        predicted_label = class_names[predicted_class[0]]

                        if predicted_label != last_stored_prediction:
                            predictions.append(predicted_label)
                            last_stored_prediction = predicted_label

                        last_prediction_time = current_time

                    # Display the prediction on the frame
                    cv2.putText(image, f"Prediction: {predicted_label}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            else:
                if no_hand_detected_time is None:
                    no_hand_detected_time = time.time()

                if no_hand_detected_time is not None and time.time() - no_hand_detected_time > no_hand_duration:
                    if predictions:
                        corrected_sentence = standardize_sentence(predictions, language)
                        st.write(f"Corrected Sentence: {corrected_sentence}")
                        engine.say(corrected_sentence)
                        engine.runAndWait()
                        
                        predictions = []
                        last_stored_prediction = None
                    no_hand_detected_time = None

            # Convert the frame back to BGR and update the frame window for display in Streamlit
            frame_window.image(image, channels='RGB')

        else:
            st.write("Recording stopped. Press 'Start Recording' to resume.")
            cap.release()
            break
         
 # Proper cleanup when stopped
        if not recording:
            cap.release()
            cv2.destroyAllWindows()
            break

# Release the Web capture when done
cap.release()
cv2.destroyAllWindows()
