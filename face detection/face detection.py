import cv2
import threading
from deepface import DeepFace

# setting the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = 0

face_match = False

img = cv2.imread("face detection\Faces\Mohamed Mostafa.png") # load the image

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


