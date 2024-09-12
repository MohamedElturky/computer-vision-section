import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Create a simple CNN model for face recognition
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')  # 4 output classes for the 4 people
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load your training data and labels here
# X_train should be the preprocessed images of shape (64, 64, 1)
# y_train should be the corresponding labels
# model.fit(X_train, y_train, epochs=10)

# After training, save the model
model.save('face_cnn_model.h5')

import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the pre-trained CNN model
model = tf.keras.models.load_model('face_cnn_model.h5')

# Define labels for each class
people = ['Bill Gates', 'Dara Khosrowshahi', 'Mark Zuckerberg', 'Sundar Pichai']

# Load the Haar Cascade for face detection
face_cascade = cv.CascadeClassifier('haar_face.xml')

# Initialize the webcam
video_capture = cv.VideoCapture(0)

def preprocess_face(roi):
    # Resize to 64x64 (as expected by the model), convert to grayscale, and normalize
    face_resized = cv.resize(roi, (64, 64))
    face_gray = cv.cvtColor(face_resized, cv.COLOR_BGR2GRAY)
    face_gray = face_gray.reshape(1, 64, 64, 1) / 255.0  # Normalize
    return face_gray

while True:
    ret, frame = video_capture.read()
    
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Loop over detected faces
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        
        # Preprocess the face region of interest
        face_input = preprocess_face(face_roi)
        
        # Predict the face using the CNN model
        predictions = model.predict(face_input)
        label_index = np.argmax(predictions)
        confidence = np.max(predictions) * 100
        person_name = people[label_index]

        # Display the name and confidence level
        label_text = f'{person_name} ({confidence:.2f}%)'

        # Draw a rectangle around the face and put text
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.putText(frame, label_text, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Display the resulting frame
    cv.imshow('Face Recognition', frame)

    # Break the loop on pressing 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
video_capture.release()
cv.destroyAllWindows()
