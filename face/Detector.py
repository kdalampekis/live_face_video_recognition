import cv2
import numpy as np
import tensorflow as tf
import time

# Load the trained model
loaded_model = tf.saved_model.load('results1')

# Define the function for making predictions on the loaded model
@tf.function(input_signature=[{'image': tf.TensorSpec(shape=(None, None, 224, 224, 3), dtype=tf.float32)}])
def predict_fn(inputs):
    logits = loaded_model(inputs)
    return {'output': tf.argmax(logits, axis=1, output_type=tf.int32)}

_MODEL_FPS = 5  # Ensure the input images are fed to the model at this fps.
_MODEL_FPS_ERROR_RANGE = 0.1  # Acceptable error range in fps.
fps, last_inference_start_time, time_per_infer = 0, 0, 0


# Open a video capture object for live video stream (use appropriate video source)
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FPS, 5)

# Initialize a list of frames
frames = []
counter = 0

# Set the desired interval between frames (in number of frames)
interval = 10

# Initialize a boolean variable to track if pill is taken
pill_taken = False

# Set the time to run the detection algorithm for
max_time = 15  # 60 seconds

start_time = time.time()
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Show the frame
    cv2.imshow('Video', frame)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Increment the counter
    counter += 1
    current_frame_start_time = time.time()
    diff = current_frame_start_time - last_inference_start_time
    if diff * _MODEL_FPS >= (1 - _MODEL_FPS_ERROR_RANGE):
        # Store the time when inference starts.
        last_inference_start_time = current_frame_start_time

        # Calculate the inference FPS
        fps = 1.0 / diff

    # If counter is a multiple of the desired interval, proceed to preprocessing and prediction
    if counter % interval == 0:
        # Preprocess the frame
        resized_frame = cv2.resize(frame, (224, 224))

        # Add the frame to a list
        frames.append(resized_frame)

    # If list is the disired length, proceed to prediction
    if len(frames) == 20:

        # Convert the list of frames to the right shape and type
        frames = np.array(frames)
        input_tensor = np.expand_dims(frames, axis=0)
        input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)

        # Make a prediction
        prediction = predict_fn({'image': input_tensor})['output'].numpy()[0]


        if prediction == 0:
            text = 'pill Taken'
            pill_taken = True
            print(text)
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            break
        else:
            # Re-initialize the list of frames
            frames = []
            counter = 0


    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q') or time.time() - start_time >= max_time:
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
if pill_taken == False:
    print("pill not Taken")

