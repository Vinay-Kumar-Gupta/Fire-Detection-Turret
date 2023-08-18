import cv2
import numpy as np
#ee

def detect_fire(frame):
    # Convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the lower and upper thresholds for the color of fire (red-orange range)
    lower_fire = np.array([0, 120, 70])
    upper_fire = np.array([20, 255, 255])
    
    # Create a mask to detect the fire color
    fire_mask = cv2.inRange(hsv_frame, lower_fire, upper_fire)
    
    # Apply morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes around potential fire regions
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # You can adjust this threshold
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    return frame

# Open the video capture
cap = cv2.VideoCapture(0)  # Replace with your video file

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    processed_frame = detect_fire(frame)
    
    cv2.imshow('Fire Detection', processed_frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
