from flask import Flask, request, jsonify
    
import librosa
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


data_dir = 'data'
classes = ['01', '02', '03', '04', '05']
X_train = []
y_train = []
X_test = []
y_test = []

n_mfcc = 13
frames = 41
n_fft = 2048
hop_length = 512

def trainModel():
    # Define the parameters for the neural network
    input_shape = (n_mfcc, frames, 1)
    num_classes = len(classes)


    # Load the audio files and extract the features
    features = []
    labels = []

    for c in classes:
        class_dir = os.path.join(data_dir, c)
        for filename in os.listdir(class_dir):
            file_path = os.path.join(class_dir, filename)
            signal, sr = librosa.load(file_path, sr=None)
            mfcc = librosa.feature.mfcc(signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
            mfcc = mfcc[:, :frames]
            mfcc = np.pad(mfcc, ((0, 0), (0, frames - mfcc.shape[1])), mode='constant')
            features.append(mfcc)
            labels.append(classes.index(c))


    # Convert the features and labels to NumPy arrays
    X = np.array(features)
    y = tf.keras.utils.to_categorical(labels, num_classes=num_classes)


    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Define the neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=16)


    # Save the trained model
    model.save('saved_models/model.h5')


def predict(file_path):
    # Define the file path and class
    # file_path = '05.wav'

    # Load the audio file and extract the features
    signal, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc = mfcc[:, :frames]
    mfcc = np.pad(mfcc, ((0, 0), (0, frames - mfcc.shape[1])), mode='constant')

    # Reshape the features to match the input shape of the model
    mfcc = np.reshape(mfcc, (1, n_mfcc, frames, 1))

    # Use the model to predict the class of the audio file
    model = tf.keras.models.load_model('saved_models/model.h5')
    pred = model.predict(mfcc)[0]
    class_idx = np.argmax(pred)
    class_name = classes[class_idx]

    if pred[class_idx] < 0.5:
        class_name = "unknown"

    return class_name

def getMetrics():
    model = tf.keras.models.load_model('saved_models/model.h5')
    loss, accuracy = model.evaluate(X_test, y_test)
    return loss, accuracy


@app.after_request
def add_cors_headers(response):
    # set the CORS headers
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Content-Type, X-Auth-Token'

    return response

@app.route("/")
def healthCheck():
    return jsonify({"message": "it works"})


@app.route("/login", methods=["POST"])
def login():
    if request.method == "POST":
        if "file" not in request.files:
            return "Error: No file provided"
        
        file = request.files["file"]
        # Check if the file is a WAV file
        # if file.mimetype != "audio/wav":
        #     return "Error: File is not a WAV file"
        
        # Save the file to disk
        
        file.save("uploads/audio.wav")

        class_name = predict("uploads/audio.wav")

        return jsonify({"message": "Success", "id": class_name})
    else:
        return jsonify({"message": "Error: Invalid request method"})



@app.route("/train", methods=["POST"])
def train():
    if request.method == "POST":
        trainModel()
        return jsonify({"message": "Success"})
    else:
        return jsonify({"message": "Error: Invalid request method"})



if __name__ == "__main__":
    app.run(debug=True)

