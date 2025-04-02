import cv2
import numpy as np
import tensorflow as tf
# Load the trained model
model = tf.saved_model.load('/Users/kostasbekis/live_face_recognition/face/results')

# Open a video capture object for live video stream (use appropriate video source)
cap = cv2.VideoCapture(0)

# Loop over frames from the live video stream
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame (resize to model input size, convert to RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.expand_dims(frame, axis=0)
    frame = frame / 255.0  # Normalize pixel values

    # Use the trained model to predict the action label (pill taking or not)
    pred = tf.keras.Model.predict(frame)
    action_label = "Pill Taking" if pred[0][0] > 0.5 else "Not Pill Taking"

    # Display the predicted action label on the video frame
    cv2.putText(frame, action_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('Live Video', frame)

    # Exit the loop on pressing 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
