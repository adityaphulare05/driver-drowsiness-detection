import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import pygame
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from pygame import mixer
import os

# Initialize pygame for sound alerts
mixer.init()
try:
    # Try to load different alert sounds
    drowsiness_alert = mixer.Sound('drowsiness_alert.wav')
    phone_alert = mixer.Sound('phone_alert.wav')
    seatbelt_alert = mixer.Sound('seatbelt_alert.wav')
    glare_alert = mixer.Sound('glare_alert.wav')
    print("All alert sounds loaded successfully.")
except:
    print("Warning: Could not load specific alert sounds. Using default alert.wav")
    try:
        default_alert = mixer.Sound('alert.wav')
        drowsiness_alert = default_alert
        phone_alert = default_alert
        seatbelt_alert = default_alert
        glare_alert = default_alert
    except:
        print("Using visual alerts only.")
        drowsiness_alert = None
        phone_alert = None
        seatbelt_alert = None
        glare_alert = None

# Load models
face_detector = dlib.get_frontal_face_detector()
try:
    landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    print("Facial landmark predictor loaded successfully.")
except:
    print("Error: Could not load shape_predictor_68_face_landmarks.dat")
    exit(1)

# Load eye state model
try:
    eye_model = load_model('models/eye_state_model.h5')
    print("Eye state model loaded successfully.")
except:
    try:
        eye_model = load_model('eye_state_model.h5')
    except:
        try:
            eye_model = load_model(r"C:\Users\sahil\Documents\eye_state_model.h5")
        except:
            print("Error: Could not load eye state model.")
            exit(1)

# Updated Constants for Better Sensitivity
EYE_AR_THRESH = 0.23  # More sensitive to eye closure
EYE_AR_CONSEC_FRAMES = 12  # Faster drowsiness detection
YAWN_THRESH = 18  # More sensitive to yawning
GLARE_THRESH = 190  # More sensitive to glare
PHONE_CONSECUTIVE_FRAMES = 8  # Faster phone detection

# Alarm states
DROWSINESS_ALARM_ON = False
PHONE_ALARM_ON = False
SEATBELT_ALARM_ON = False
GLARE_ALARM_ON = False

# Counters
COUNTER = 0
phone_detection_counter = 0

# Pre-allocate arrays
left_eye_points = np.array([36, 37, 38, 39, 40, 41])
right_eye_points = np.array([42, 43, 44, 45, 46, 47])

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def detect_eyeblink(gray, rect):
    shape = landmark_predictor(gray, rect)
    shape_np = face_utils.shape_to_np(shape)
    leftEye = shape_np[left_eye_points]
    rightEye = shape_np[right_eye_points]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    return (leftEAR + rightEAR) / 2.0, leftEye, rightEye

def detect_yawn(shape):
    mouth = shape[48:68]
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    return (A + B) / 2.0

def detect_gaze(eye_region):
    eye_center = np.mean(eye_region, axis=0)
    left_corner = eye_region[0]
    right_corner = eye_region[3]
    left_diff = left_corner[0] - eye_center[0]
    right_diff = eye_center[0] - right_corner[0]
    if left_diff > right_diff + 4:
        return "Right"
    elif right_diff > left_diff + 4:
        return "Left"
    else:
        return "Center"

def enhance_phone_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)
    edged = cv2.Canny(morphed, 50, 150)
    dilated = cv2.dilate(edged, kernel, iterations=1)
    return dilated, gray

def detect_phone_enhanced(image):
    """
    Enhanced phone detection with better accuracy
    """
    global phone_detection_counter, PHONE_ALARM_ON
    
    # Multiple detection strategies
    strategies = []
    
    # Strategy 1: Edge-based detection
    processed1, gray1 = enhance_phone_detection(image)
    contours1, _ = cv2.findContours(processed1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Strategy 2: Color-based detection (common phone colors)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Black/dark gray phones
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 80])
    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
    
    # Silver/white phones
    lower_light = np.array([0, 0, 200])
    upper_light = np.array([180, 50, 255])
    light_mask = cv2.inRange(hsv, lower_light, upper_light)
    
    color_mask = cv2.bitwise_or(dark_mask, light_mask)
    
    # Clean up color mask
    kernel = np.ones((3, 3), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    
    contours2, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    phone_detected = False
    phone_bbox = None
    best_confidence = 0
    
    # Check contours from both strategies
    all_contours = list(contours1) + list(contours2)
    
    for contour in all_contours:
        area = cv2.contourArea(contour)
        if area < 500:  # Increased minimum area
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        
        # Phone-like aspect ratios (most phones are between 1.5:1 and 2:1)
        if 0.5 <= aspect_ratio <= 2.5:
            rect_area = w * h
            extent = area / float(rect_area) if rect_area > 0 else 0
            
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / float(hull_area) if hull_area > 0 else 0
            
            # Calculate confidence score
            confidence = 0
            if 0.6 <= aspect_ratio <= 2.2:
                confidence += 1
            if extent > 0.6:
                confidence += 1
            if solidity > 0.7:
                confidence += 1
            if 1000 <= area <= 30000:  # Reasonable phone size
                confidence += 1
            
            if confidence > best_confidence and confidence >= 2:
                best_confidence = confidence
                phone_detected = True
                phone_bbox = (x, y, w, h)
    
    # Consecutive frame validation
    if phone_detected:
        phone_detection_counter += 1
        if phone_detection_counter >= PHONE_CONSECUTIVE_FRAMES:
            if not PHONE_ALARM_ON:
                if phone_alert:
                    phone_alert.play()
                PHONE_ALARM_ON = True
                print("ALERT: Phone usage detected while driving!")
            
            # Draw detection with confidence
            if phone_bbox:
                x, y, w, h = phone_bbox
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(image, f"PHONE DETECTED ({best_confidence}/4)", 
                           (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        phone_detection_counter = max(0, phone_detection_counter - 2)
        if phone_detection_counter == 0:
            PHONE_ALARM_ON = False
    
    return phone_detected and phone_detection_counter >= PHONE_CONSECUTIVE_FRAMES, phone_bbox

def detect_seatbelt_enhanced(image):
    """
    Enhanced seatbelt detection using multiple approaches
    """
    global SEATBELT_ALARM_ON
    
    # Method 1: Color-based detection (gray/black seatbelts)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define ranges for seatbelt colors
    lower_gray1 = np.array([0, 0, 50])
    upper_gray1 = np.array([180, 50, 150])
    
    lower_gray2 = np.array([0, 0, 0])
    upper_gray2 = np.array([180, 255, 100])
    
    mask1 = cv2.inRange(hsv, lower_gray1, upper_gray1)
    mask2 = cv2.inRange(hsv, lower_gray2, upper_gray2)
    color_mask = cv2.bitwise_or(mask1, mask2)
    
    # Method 2: Edge detection for seatbelt straps
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Combine methods
    combined_mask = cv2.bitwise_or(color_mask, edges)
    
    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    seatbelt_detected = False
    seatbelt_confidence = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area
        if 500 < area < 20000:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            # Seatbelt-like characteristics
            if aspect_ratio > 4.0 or aspect_ratio < 0.25:  # Very horizontal or vertical
                # Calculate confidence score
                rect_area = w * h
                extent = area / float(rect_area) if rect_area > 0 else 0
                
                # Check for line-like features
                if extent > 0.3:
                    seatbelt_confidence += 1
                    
                    # Draw potential seatbelt
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(image, f"SEATBELT {seatbelt_confidence}", 
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Consider seatbelt detected if we found multiple good candidates
    seatbelt_detected = seatbelt_confidence >= 2
    
    # Alarm logic
    if not seatbelt_detected and not SEATBELT_ALARM_ON:
        if seatbelt_alert:
            seatbelt_alert.play()
        SEATBELT_ALARM_ON = True
        print("ALERT: Seatbelt not detected! Please wear your seatbelt.")
    elif seatbelt_detected:
        SEATBELT_ALARM_ON = False
    
    return seatbelt_detected

def detect_glare_enhanced(image, face_region=None):
    """
    Enhanced glare detection for sunlight and headlights
    """
    global GLARE_ALARM_ON
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Analyze different regions
    if face_region is not None:
        fx, fy, fw, fh = face_region
        # Check multiple regions around face
        regions = [
            (max(0, fx - fw), max(0, fy - fh), fw * 2, fh * 2),  # Expanded area
            (fx, fy, fw, fh),  # Face area itself
            (max(0, fx - fw//2), max(0, fy - fh//2), fw * 2, fh//2)  # Upper region
        ]
    else:
        regions = [(0, 0, gray.shape[1], gray.shape[0])]
    
    glare_detected = False
    max_glare_intensity = 0
    
    for region in regions:
        x, y, w, h = region
        if w <= 0 or h <= 0:
            continue
            
        region_gray = gray[y:y+h, x:x+w]
        if region_gray.size == 0:
            continue
        
        # Calculate brightness statistics
        avg_brightness = np.mean(region_gray)
        max_brightness = np.max(region_gray)
        
        # Find bright spots
        _, bright_spots = cv2.threshold(region_gray, GLARE_THRESH, 255, cv2.THRESH_BINARY)
        
        # Analyze bright regions
        bright_pixels = np.sum(bright_spots == 255)
        total_pixels = bright_spots.size
        
        if total_pixels > 0:
            bright_ratio = bright_pixels / total_pixels
            
            # Glare detection criteria
            if (max_brightness > GLARE_THRESH + 20 and 
                bright_ratio > 0.03 and 
                avg_brightness > 100):
                
                glare_detected = True
                max_glare_intensity = max(max_glare_intensity, max_brightness)
                
                # Draw glare warning
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(image, f"GLARE: {int(max_brightness)}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                break
    
    # Alarm logic with intensity-based triggering
    if glare_detected and not GLARE_ALARM_ON:
        if glare_alert:
            glare_alert.play()
        GLARE_ALARM_ON = True
        print(f"ALERT: Strong glare detected! Intensity: {max_glare_intensity}")
    elif not glare_detected:
        GLARE_ALARM_ON = False
    
    return glare_detected

def detect_hand_near_face(face_rect, phone_bbox):
    if phone_bbox is None:
        return False
        
    fx, fy, fw, fh = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
    px, py, pw, ph = phone_bbox
    
    face_center = np.array([fx + fw/2, fy + fh/2])
    phone_center = np.array([px + pw/2, py + ph/2])
    distance = np.linalg.norm(face_center - phone_center)
    
    return distance < fw * 2

def trigger_drowsiness_alarm():
    global DROWSINESS_ALARM_ON
    if not DROWSINESS_ALARM_ON and drowsiness_alert:
        drowsiness_alert.play()
        DROWSINESS_ALARM_ON = True

def stop_drowsiness_alarm():
    global DROWSINESS_ALARM_ON
    DROWSINESS_ALARM_ON = False

def draw_safety_alerts(frame, alerts):
    """
    Draw comprehensive safety alerts on the frame
    """
    alert_y = 400
    for i, (alert_text, color) in enumerate(alerts):
        cv2.putText(frame, alert_text, (10, alert_y + (i * 30)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Draw overall safety status
    if alerts:
        cv2.putText(frame, "SAFETY ALERTS ACTIVE!", (10, 350), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "ALL SYSTEMS NORMAL", (10, 350), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

def main():
    global COUNTER, DROWSINESS_ALARM_ON
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Starting Enhanced Driver Safety System...")
    print("Monitoring: Drowsiness, Phone Usage, Seatbelt, Glare")
    print("Press 'q' to quit")
    
    frame_count = 0
    ear = 0.0
    mar = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 2 != 0:
            continue
            
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
        
        # Detect all safety features
        phone_detected, phone_bbox = detect_phone_enhanced(display_frame)
        seatbelt_detected = detect_seatbelt_enhanced(display_frame)
        
        rects = face_detector(small_gray, 0)
        
        eyes_closed = False
        yawning = False
        looking_away = False
        phone_being_used = False
        glare_detected = False
        
        ear = 0.0
        mar = 0.0
        face_region_for_glare = None
        
        # Process each detected face
        for rect in rects:
            rect_scaled = dlib.rectangle(
                rect.left() * 2, rect.top() * 2,
                rect.right() * 2, rect.bottom() * 2
            )
            
            landmarks = landmark_predictor(gray, rect_scaled)
            shape = face_utils.shape_to_np(landmarks)
            
            ear, leftEye, rightEye = detect_eyeblink(gray, rect_scaled)
            
            # Drowsiness detection
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    eyes_closed = True
                    trigger_drowsiness_alarm()
            else:
                COUNTER = 0
                stop_drowsiness_alarm()
                
            mar = detect_yawn(shape)
            if mar > YAWN_THRESH:
                yawning = True
                trigger_drowsiness_alarm()
                
            # Gaze detection
            if ear > EYE_AR_THRESH:
                left_gaze = detect_gaze(shape[left_eye_points])
                right_gaze = detect_gaze(shape[right_eye_points])
                if left_gaze != "Center" or right_gaze != "Center":
                    looking_away = True
            
            # Store face region for glare detection
            face_region_for_glare = (rect_scaled.left(), rect_scaled.top(), 
                                   rect_scaled.width(), rect_scaled.height())
            
            # Check phone usage near face
            if phone_detected:
                phone_being_used = detect_hand_near_face(rect_scaled, phone_bbox)
            
            # Draw face landmarks
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(display_frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(display_frame, [rightEyeHull], -1, (0, 255, 0), 1)
            
            mouth = shape[48:68]
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(display_frame, [mouthHull], -1, (0, 255, 0), 1)
        
        # Glare detection (around face region)
        if face_region_for_glare:
            glare_detected = detect_glare_enhanced(display_frame, face_region_for_glare)
        else:
            glare_detected = detect_glare_enhanced(display_frame)
        
        # Display comprehensive status
        status_y = 30
        cv2.putText(display_frame, f"EAR: {ear:.2f}", (10, status_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        # Safety status indicators
        status_items = [
            (f"Eyes: {'Closed' if eyes_closed else 'Open'}", 
             (0, 0, 255) if eyes_closed else (0, 255, 0)),
            (f"Yawn: {'Yes' if yawning else 'No'}", 
             (0, 0, 255) if yawning else (0, 255, 0)),
            (f"Phone: {'USING!' if phone_being_used else 'Detected' if phone_detected else 'No'}", 
             (0, 0, 255) if phone_being_used else (255, 165, 0) if phone_detected else (0, 255, 0)),
            (f"Seatbelt: {'ON' if seatbelt_detected else 'OFF'}", 
             (0, 255, 0) if seatbelt_detected else (0, 0, 255)),
            (f"Glare: {'DETECTED' if glare_detected else 'No'}", 
             (0, 255, 255) if glare_detected else (0, 255, 0))
        ]
        
        for i, (text, color) in enumerate(status_items):
            cv2.putText(display_frame, text, (10, status_y + 25 + (i * 25)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # Active alarms display using the new alert system
        alerts = []
        if DROWSINESS_ALARM_ON:
            alerts.append(("DROWSINESS DETECTED!", (0, 0, 255)))
        if PHONE_ALARM_ON:
            alerts.append(("PHONE USAGE DETECTED!", (0, 0, 255)))
        if SEATBELT_ALARM_ON:
            alerts.append(("SEATBELT NOT WORN!", (0, 0, 255)))
        if GLARE_ALARM_ON:
            alerts.append(("STRONG GLARE DETECTED!", (0, 255, 255)))
        
        # Draw safety alerts
        draw_safety_alerts(display_frame, alerts)
        
        cv2.imshow("Enhanced Driver Safety System", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Safety system closed.")

if __name__ == "__main__":
    main()