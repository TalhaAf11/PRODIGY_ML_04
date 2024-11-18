import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

# Initialize MediaPipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Load the trained gesture recognition model
model = tf.keras.models.load_model("gesture_model.h5")

# Initialize OpenCV video capture (use webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror view
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB (MediaPipe uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the hand
            for landmark in landmarks.landmark:
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            # Extract the hand region and preprocess for classification
            # Crop the image around the detected hand
            x_min, x_max = min([l.x for l in landmarks.landmark]), max([l.x for l in landmarks.landmark])
            y_min, y_max = min([l.y for l in landmarks.landmark]), max([l.y for l in landmarks.landmark])
            cropped_hand = frame[int(y_min * h):int(y_max * h), int(x_min * w):int(x_max * w)]

            # Resize cropped hand image to model input size
            cropped_hand_resized = cv2.resize(cropped_hand, (224, 224)) / 255.0
            cropped_hand_expanded = np.expand_dims(cropped_hand_resized, axis=0)

            # Predict gesture class using the trained model
            prediction = model.predict(cropped_hand_expanded)
            gesture_class = np.argmax(prediction)

            # Display the predicted gesture on the frame
            cv2.putText(frame, f"Gesture: {gesture_class}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame with hand landmarks and predicted gesture
    cv2.imshow("Hand Gesture Recognition", frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
