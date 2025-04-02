import numpy as np
import cv2
import tensorflow as tf
import argparse
import sys
import time
import cv2
import tflite

# Visualization parameters
_ROW_SIZE = 20  # pixels
_LEFT_MARGIN = 24  # pixels
_TEXT_COLOR = (0, 0, 255)  # red
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_MODEL_FPS = 5  # Ensure the input images are fed to the model at this fps.
_MODEL_FPS_ERROR_RANGE = 0.1  # Acceptable error range in fps.

# Load the trained model
loaded_model = tf.saved_model.load('results')

# Define the function for making predictions on the loaded model
@tf.function(input_signature=[{'image': tf.TensorSpec(shape=(None, None, 224, 224, 3), dtype=tf.float32)}])
def predict_fn(inputs):
    logits = loaded_model(inputs)
    return {'output': tf.argmax(logits, axis=1, output_type=tf.int32)}

# Variables to calculate FPS
fps, last_inference_start_time, time_per_infer = 0, 0, 0
text = 'kostas'
counter = 0
frames = []
interval = 1
# Start capturing video input from the camera
cap = cv2.VideoCapture(0)


# Continuously capture images from the camera and run inference
while cap.isOpened():
    ret, frame = cap.read()
    # Show the frame
    cv2.imshow('Video', frame)
    counter += 1

    # Mirror the image
    frame = cv2.flip(frame, 1)

    # Ensure that frames are feed to the model at {_MODEL_FPS} frames per second
    # as required in the model specs.
    current_frame_start_time = time.time()
    diff = current_frame_start_time - last_inference_start_time
    if diff * _MODEL_FPS >= (1 - _MODEL_FPS_ERROR_RANGE):
      # Store the time when inference starts.
      last_inference_start_time = current_frame_start_time

      # Calculate the inference FPS
      fps = 1.0 / diff

      # Convert the frame to RGB as required by the TFLite model.
      frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      if counter % interval == 0:
          # Preprocess the frame
          resized_frame = cv2.resize(frame, (224, 224))

          # Add the frame to a list
          frames.append(resized_frame)
    # Decide whether the person is taking a pill or not.
    if len(frames) == 20:

        # Convert the list of frames to the right shape and type
        frames = np.array(frames)
        input_tensor = np.expand_dims(frames, axis=0)
        input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)

        # Make a prediction
        prediction = predict_fn({'image': input_tensor})['output'].numpy()[0]

        # Display the prediction on the frame
        if prediction == 0:
            text = 'Not taking pill'
        else:
            text = 'Taking pill'

        print(text)
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Re-initialize the list of frames
    frames = []
    counter = 0

    # Exit on 'q' keypress.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream.
cap.release()
cv2.destroyAllWindows()
