import cv2
import numpy as np
import face_recognition
import os

# Define constants
FAMILY_FOLDER = "face detection/Faces"
TOLERANCE = 0.6

# Load known face encodings and names
known_face_encodings = []
known_face_names = []

for filename in os.listdir(FAMILY_FOLDER):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        family_image = face_recognition.load_image_file(os.path.join(FAMILY_FOLDER, filename))
        family_encoding = face_recognition.face_encodings(family_image)[0]
        known_face_encodings.append(family_encoding)
        known_face_names.append(os.path.splitext(filename)[0])

# Initialize face detection cascade classifier
face_detect = cv2.CascadeClassifier("myvenv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")

# Initialize video capture
vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    if not ret:
        break

    # Detect faces in the frame
    faces = face_detect.detectMultiScale(frame, 1.3, 5)

    for (x, y, w, h) in faces:
        # Extract face ROI
        face_only = frame[y:y + h, x:x + w]

        # Convert face ROI to RGB
        rgb_face = cv2.cvtColor(face_only, cv2.COLOR_BGR2RGB)

        # Get face encoding
        face_encodings = face_recognition.face_encodings(rgb_face)

        if face_encodings:
            face_encoding = face_encodings[0]

            # Compare face encoding with known face encodings
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=TOLERANCE)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                # Draw rectangle around the face with the recognized name
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            else:
                name = "No match"
                # Draw rectangle around the face with the recognized name
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            

    # Resize the frame and add borders
    frame = cv2.resize(frame, (1000, 800))

    # Display the output
    cv2.imshow("Smart home camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()