import librosa
import numpy as np
import os
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Define the dataset directory and the classes
DATASET_DIR = 'data'
CLASSES = ['01', '02']

# Define the number of MFCC coefficients
NUM_MFCC = 20


# Load the data
def load_data():
    # Create empty lists for the data and the labels
    data = []
    labels = []

    # Loop over the classes
    for class_label in CLASSES:
        # Get the directory path for the class
        class_dir = os.path.join(DATASET_DIR, class_label)

        # Loop over the audio files in the class directory
        for filename in os.listdir(class_dir):
            # Load the audio file
            filepath = os.path.join(class_dir, filename)
            signal, sr = librosa.load(filepath)

            # Extract the MFCC features
            mfccs = librosa.feature.mfcc(signal, sr, n_mfcc=NUM_MFCC)

            # Append the MFCC features and the label to the data and labels lists
            data.append(mfccs.T)
            labels.append(CLASSES.index(class_label))

    # Convert the data and labels to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    return x_train, y_train, x_test, y_test


# Define the neural network model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(None, NUM_MFCC)),
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# Train the model
def train_model(x_train, y_train, x_test, y_test):
    # Build the model
    model = build_model()

    # Train the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=32)

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(x_test, y_test)

    model.save('saved_model/voice_auth')

    print(f'Test loss: {loss:.3f}, accuracy: {accuracy:.3f}')


# Load the data
x_train, y_train, x_test, y_test = load_data()

# Train the model
train_model(x_train, y_train, x_test, y_test)


# # Define the number of MFCC coefficients
# NUM_MFCC = 20

# # Load the audio sample
# audio_path = 'path/to/audio/sample'
# signal, sr = librosa.load(audio_path)

# # Extract the MFCC features
# mfccs = librosa.feature.mfcc(signal, sr, n_mfcc=NUM_MFCC)

# # Reshape the MFCC features to match the input shape of the model
# mfccs = np.expand_dims(mfccs.T, axis=0)

# # Load the trained model
# model_path = 'saved_model/voice_auth'
# model = tf.keras.models.load_model(model_path)

# # Use the model to classify the audio sample
# class_probabilities = model.predict(mfccs)
# class_index = np.argmax(class_probabilities)
# predicted_class = CLASSES[class_index]

# print(f'The predicted class is {predicted_class}.')