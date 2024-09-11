import cv2
import threading
from deepface import DeepFace

# setting the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = 0

face_match = False

#img = cv2.imread("face detection\Faces\Mohamed Mostafa.png") # load the image
img = cv2.imread(r'C:\Users\mirol\Documents\GitHub\computer-vision-section\face detection\Faces\Mohamed Mostafa.png') # load the image

Cascade =cv2.CascadeClassifier('haar_face.xml')
people=['Bill Gates','Dara Khosrowshani','Mark Zuckerberg','Sudan Pichai']
face_recognizer= cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

def detect_and_recognize_faces(img, gray, faces_rect, face_recognizer, people):
    face_found = False

    for (x, y, w, h) in faces_rect:
        face_found = True
        faces_roi = gray[y:y+h, x:x+w]
        label, confidence_level = face_recognizer.predict(faces_roi)
        print(f'label={people[label]}:{confidence_level}')
        cv.putText(img, str(people[label]), (x-20, y-5), cv.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), thickness=2)
        cv.rectangle(img, (x, y), (x+w, y+h), (100, 0, 255), thickness=3)

    return face_found


def check_face(frame):
    global face_match

    try:
        if DeepFace.verify(frame, img.copy())['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False

while True:
    ret, frame = cap.read()

    if ret:
        if count % 30 == 0: #check every 30 frams
            try:
                threading.Thread(target = check_face, args = (frame,)).start()

            except ValueError:
                pass

        count += 1

        if face_match:
            cv2.putText(frame, "Face Matched", (20, 450), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "No Face Matched", (20, 450), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame)


    key =  cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


