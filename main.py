
import cv2
import face_recognition
import time
from io import BytesIO
from time import sleep
from picamera import PiCamera
from PIL import Image
from multiprocessing import Pool, TimeoutError

# sudo apt install libatlas3-base libwebp6 libtiff5 libjasper1 libilmbase23 libopenexr23 libavcodec58 libavformat58 libavutil56 libswscale5 libgtk-3-0 libpangocairo-1.0-0 libpango-1.0-0 libatk1.0-0 libcairo-gobject2 libcairo2 libgdk-pixbuf2.0-0 libqtgui4 libqt4-test libqtcore4
# sudo pip3 install opencv-python
N = 5

cap = cv2.VideoCapture(0)

def preprocess(frame):
    # convert to gray to boost performance
    x = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x = cv2.resize(x, (0, 0), fx=1 / N, fy=1 / N)
    return x

def display(frame):
    # Display the resulting image
    cv2.imshow('Video', frame)


def draw_face_location(frame, face_locations):
    for (top, right, bottom, left) in face_locations:
        print(f'[FACE] @top={top} right={right} bottom={bottom} left={left}')
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= N
        right *= N
        bottom *= N
        left *= N
        # Draw a box around the face
        frame = cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 3)

    return frame

def extract_faces(frame, face_locations):
    faces = []
    for (top, right, bottom, left) in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= N
        right *= N
        bottom *= N
        left *= N
        # Draw a box around the face
        print(frame.shape)
        face = frame[left:right,top:bottom,:]
        
        faces.append(face)

    return faces

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    start = time.time()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    x = preprocess(frame)
    face_locations = face_recognition.face_locations(x)
    frame = draw_face_location(frame, face_locations)
    # faces = extract_faces(frame, face_locations)
    print(f"[ELAPSED] {time.time() - start:.2f}s")
    if len(faces) > 0:
        print(faces[0].shape)
        display(faces[0])
    else:
        display(frame)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
