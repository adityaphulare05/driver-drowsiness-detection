import cv2
import dlib
import numpy as np
import os
from imutils import face_utils

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Create directories if they don't exist
os.makedirs("data/eyes/open", exist_ok=True)
os.makedirs("data/eyes/closed", exist_ok=True)

def extract_eye_region(face_landmarks, gray, eye_points):
    # Extract eye region
    eye_region = np.array([(face_landmarks.part(point).x, face_landmarks.part(point).y) for point in eye_points])
    
    # Get the bounding box of the eye region
    x, y, w, h = cv2.boundingRect(eye_region)
    
    # Extract eye image
    eye_image = gray[y:y+h, x:x+w]
    
    # Resize to standard size
    eye_image = cv2.resize(eye_image, (24, 24))
    
    return eye_image

cap = cv2.VideoCapture(0)
open_count = 0
closed_count = 0

print("Press 'o' to capture open eye, 'c' for closed eye, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    
    for face in faces:
        landmarks = predictor(gray, face)
        
        # Extract left eye (points 36-41)
        left_eye = extract_eye_region(landmarks, gray, range(36, 42))
        
        # Extract right eye (points 42-47)
        right_eye = extract_eye_region(landmarks, gray, range(42, 48))
        
        # Display eye images
        cv2.imshow("Left Eye", left_eye)
        cv2.imshow("Right Eye", right_eye)
        
        # Draw face landmarks
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('o'):  # Save open eye
        cv2.imwrite(f"data/eyes/open/open_{open_count}.jpg", left_eye)
        cv2.imwrite(f"data/eyes/open/open_{open_count+1}.jpg", right_eye)
        open_count += 2
        print(f"Saved {open_count} open eye images")
        
    elif key == ord('c'):  # Save closed eye
        cv2.imwrite(f"data/eyes/closed/closed_{closed_count}.jpg", left_eye)
        cv2.imwrite(f"data/eyes/closed/closed_{closed_count+1}.jpg", right_eye)
        closed_count += 2
        print(f"Saved {closed_count} closed eye images")
        
    elif key == ord('q'):  # Quit
        break

cap.release()
cv2.destroyAllWindows()