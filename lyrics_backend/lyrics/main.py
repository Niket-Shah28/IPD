import cv2
import numpy as np
import time
import tensorflow as tf
from keras import models

from collections import Counter

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

with open('emotion.json', 'r') as json_file:
    loaded_model_json = json_file.read()
emotion_model = models.model_from_json(loaded_model_json)
emotion_model.load_weights("emotion_model.h5")
print("Loaded model from disk")

import json
from keras.models import model_from_json

with open('your_model.json', 'r') as json_file:
    loaded_model_json = json.load(json_file)
emotion_model = model_from_json(loaded_model_json)

# Load weights if separate (optional)
emotion_model.load_weights("your_model.h5")

print("Loaded model from disk")

cap = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

capture_duration = 30  
start_time = time.time()

emotion_counter = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (1280, 720))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

        emotion_prediction = emotion_model.predict(cropped_img)
        emotion_index = int(np.argmax(emotion_prediction))
        emotion_label = emotion_dict[emotion_index]

        emotion_counter.append(emotion_label)

        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if time.time() - start_time > capture_duration:
        break

cap.release()
cv2.destroyAllWindows()

most_frequent_emotion = Counter(emotion_counter).most_common(1)[0][0]
print("The most frequent emotion during the capture was:", most_frequent_emotion)

import librosa
import numpy as np
import pandas as pd
import os
import json
import joblib

def extract_features_from_audio(audio_path):
    y, sr = librosa.load(audio_path, duration=30) 
    S = np.abs(librosa.stft(y))
    
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    poly_features = librosa.feature.poly_features(S=S, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    harmonic = librosa.effects.harmonic(y)
    percussive = librosa.effects.percussive(y)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_delta = librosa.feature.delta(mfcc)
    
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    frames_to_time = librosa.frames_to_time(onset_frames[:20], sr=sr)
    
    features = {
        'song_name': os.path.basename(audio_path),
        'tempo': float(tempo),
        'total_beats': sum(beats),
        'average_beats': np.average(beats),
        'chroma_stft_mean': np.mean(chroma_stft),
        'chroma_cq_mean': np.mean(chroma_cq),
        'chroma_cens_mean': np.mean(chroma_cens),
        'melspectrogram_mean': np.mean(melspectrogram),
        'mfcc_mean': np.mean(mfcc),
        'mfcc_delta_mean': np.mean(mfcc_delta),
        'rms_mean': np.mean(rms),
        'cent_mean': np.mean(cent),
        'spec_bw_mean': np.mean(spec_bw),
        'contrast_mean': np.mean(contrast),
        'rolloff_mean': np.mean(rolloff),
        'poly_mean': np.mean(poly_features),
        'tonnetz_mean': np.mean(tonnetz),
        'zcr_mean': np.mean(zcr),
        'harm_mean': np.mean(harmonic),
        'perc_mean': np.mean(percussive),
        'frame_mean': np.mean(frames_to_time),
    }
    
    return features


audio_path = "./Ae Dil Hai Mushkil Title Track(PagalWorld.com.sb).mp3"  
features = extract_features_from_audio(audio_path)
features_df = pd.DataFrame([features])
selected_features = features_df.drop('song_name', axis=1)  

svm_classifier = joblib.load('svm_model.pkl')

scaler = joblib.load('scaler.pkl')

selector = joblib.load('feature_selector.pkl')
new_data_selected = selector.transform(selected_features)
new_data_scaled = scaler.transform(new_data_selected)
new_predictions = svm_classifier.predict(new_data_scaled)

print("Predictions for new data:")
print(new_predictions)

print(features)

emotion_to_final_mood = {
    "Angry": "angry",
    "Disgusted": "angry",  
    "Fearful": "sad",
    "Happy": "happy",
    "Neutral": "calm",
    "Sad": "sad",
    "Surprised": "calm" 
}
video_mood = emotion_to_final_mood.get(most_frequent_emotion, "calm")

if video_mood == new_predictions:
    final_mood = video_mood
else:
    final_mood = new_predictions  

print("Final combined mood:", final_mood)

final_mood='angry'
import pandas as pd

csv_file_path = "./Main_df.csv"
df = pd.read_csv(csv_file_path)
# print(df)
final_mood=final_mood[0:]
final_recommendation=[]
for index, row in df.iterrows():
    # print(row)
    # print(row['VA'])
    # Check if the 'VA' column value in the current row matches the final mood
    if row["VA"] == final_mood:
        # print(row['VA'])
        # If the condition is true, append the corresponding song name to the final recommendations list
        final_recommendation.append(row["song_name"])
# matching_songs = df["VA"] == final_mood
# print(final_recommendation)
# song_names = matching_songs["song_names"].tolist()

print("Songs with mood", final_mood, ":", final_recommendation)






