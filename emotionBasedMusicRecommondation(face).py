#python and machinelearning modules
#emotion based music recommondation system using face emotion recognition system 




import cv2
from fer import FER
import os
import random
from googleapiclient.discovery import build
import time

# Load the pre-trained emotion detection model
emotion_detector = FER()

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Google API credentials
YOUTUBE_API_KEY = 'AIzaSyCF6mjOQDNa8OUtWvWd60Fz9gtzkOo-sQ8'

youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Emotion-music mapping

# Emotion-music mapping
emotion_to_query = {
    'happy': 'happy song',
    'sad': 'sad song',
    'angry': 'angry song',
    'neutral': 'neutral background music',
    'surprised': 'surprise song',
    'fearful': 'fearful music',
    'disgusted': 'disgusted song',
    'confused': 'confused song',
    'bored': 'boredom relief song',
    'excited': 'exciting party song'
    # Add more emotions and corresponding queries here
}

# Initialize the last_opened_time
last_opened_time = 0

def search_youtube(query):
    search_response = youtube.search().list(
        q=query,
        type='video',
        part='id',
        maxResults=10
    ).execute()
    
    videos = search_response.get('items', [])
    if videos:
        return random.choice(videos)['id']['videoId']
    return None

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        emotions = emotion_detector.detect_emotions(face)

        if emotions:
            dominant_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
            print(f"Dominant Emotion: {dominant_emotion}")

            query = emotion_to_query.get(dominant_emotion, 'background music')

            # Check if enough time has passed since the last tab was opene
            current_time = time.time()
            if current_time - last_opened_time >= 300:  # 300 seconds = 5 minutes
                track_id = search_youtube(query)
                if track_id:
                    youtube_url = f'https://www.youtube.com/watch?v={track_id}'
                    os.system(f'start {youtube_url}')  # Open YouTube video in default browser
                    last_opened_time = current_time  # Update the last opened time

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
