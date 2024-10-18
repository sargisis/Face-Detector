import cv2
import numpy as np

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the initial focal length (in pixels) and the average face height (in cm)
FOCAL_LENGTH = 500  # Adjust this based on your camera's specifications
AVERAGE_FACE_HEIGHT_CM = 14.0  # Average height of a human face in cm

def estimate_distance(face_height_px, focal_length=FOCAL_LENGTH):
    if face_height_px > 0:
        return (focal_length * AVERAGE_FACE_HEIGHT_CM) / face_height_px
    return None

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Estimate distance using the height of the detected face
        distance = estimate_distance(h)
        if distance:
            # Display the distance and face size information
            cv2.putText(frame, f"Distance: {distance:.2f} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Height (px): {h}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display instructions for calibration
    cv2.putText(frame, "Press 'c' to calibrate focal length", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Face Distance Estimation', frame)

    # Capture keyboard input
    key = cv2.waitKey(1) & 0xFF

    # Exit on pressing 'q'
    if key == ord('q'):
        break
    # Calibrate focal length on pressing 'c'
    elif key == ord('c'):
        # Assuming a known distance (e.g., 100 cm) and average face height (14 cm)
        known_distance_cm = float(input("Enter the known distance (cm): "))
        known_height_px = int(input("Enter the detected face height in pixels: "))
        if known_height_px > 0:
            FOCAL_LENGTH = (known_height_px * known_distance_cm) / AVERAGE_FACE_HEIGHT_CM
            print(f"New focal length set: {FOCAL_LENGTH:.2f} pixels")

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
