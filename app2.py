import os
import pickle
import numpy as np
import wave
import librosa
import speech_recognition as sr
import pyaudio
from scipy.io import wavfile
import soundfile as sf
from sklearn.mixture import GaussianMixture
from flask import Flask, request, jsonify

r = sr.Recognizer()

app = Flask(__name__)

def mkdir(parent_dir, username):
    if not os.path.exists(f"{parent_dir}/{username}"):
        os.makedirs(f"{parent_dir}/{username}")


# Function to record voice samples for enrollment
@app.route('/enroll', methods=['POST'])
def enroll():
    username = request.form['username']
    mkdir("enrolls", username)
    num_samples = int(request.form['num_samples'])
    for i in range(num_samples):
        file = request.files[f"sample{i}"]
        filepath = f"enrolls/{username}/sample{i}.wav"
        file.save(filepath)
    return jsonify({'message': f"Enrolled {num_samples} samples for user {username}"})

# Function to train GMM classifier for speaker recognition
@app.route('/train', methods=['POST'])
def train():
    username = request.form['username']
    # Read voice samples from files
    # Load the audio files and preprocess the voice samples
    audio_1, sr_1 = librosa.load(f"enrolls/{username}/sample0.wav", sr=None)
    audio_2, sr_2 = librosa.load(f"enrolls/{username}/sample1.wav", sr=None)
    audio_3, sr_3 = librosa.load(f"enrolls/{username}/sample2.wav", sr=None)

    # Resample the audio to a fixed sample rate
    sr = 16000
    audio_1 = librosa.resample(audio_1, sr_1, sr)
    audio_2 = librosa.resample(audio_2, sr_2, sr)
    audio_3 = librosa.resample(audio_3, sr_3, sr)

    # Extract MFCC features from the preprocessed voice samples
    n_mfcc = 20
    mfcc_1 = librosa.feature.mfcc(audio_1, sr=sr, n_mfcc=n_mfcc)
    mfcc_2 = librosa.feature.mfcc(audio_2, sr=sr, n_mfcc=n_mfcc)
    mfcc_3 = librosa.feature.mfcc(audio_3, sr=sr, n_mfcc=n_mfcc)

    # Stack the MFCC features together
    X = np.vstack((mfcc_1, mfcc_2, mfcc_3)).T

    # Train a GMM on the preprocessed voice samples
    n_components = 16
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(X)

    if not os.path.exists(f"models/{username}"):
        os.makedirs(f"models/{username}")
    with open(f"models/{username}/model.pkl", 'wb') as f:
        pickle.dump(gmm, f)

    return jsonify({'message': 'Training successful'})

@app.route('/recognize', methods=['POST'])
def recognize():
    username = request.form['username']
    
    # Load the trained GMM model
    with open(f"models/{username}/model.pkl", 'rb') as f:
        gmm = pickle.load(f)
    
    # Read the uploaded audio file
    file = request.files["test"]
    filepath = f"enrolls/{username}/test.wav"
    file.save(filepath)

    test, sr_1 = librosa.load(f"enrolls/{username}/test.wav", sr=None)

    sr = 16000
    test = librosa.resample(test, sr_1, sr)

    n_mfcc = 20
    mfcc = librosa.feature.mfcc(test, sr=sr, n_mfcc=n_mfcc)

    X = np.vstack((mfcc, mfcc, mfcc)).T
    # preprocessed_sample = mfcc.reshape(1, -1)
    
    score = gmm.score(X)

    print(f"Speaker: {username}, Score: {score}")
    
    # Check if the speaker is recognized
    if score > -100:
        return jsonify({'message': f"Welcome {username}!"})
    else:
        return jsonify({'message': "Speaker not recognized"})



if __name__ == "__main__":
    app.run(debug=True)
