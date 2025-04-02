import cv2
import time

# initialize the camera
cap = cv2.VideoCapture(0)

# time organizer
max_time = 2  # 60 seconds
counter = 0
start_time = time.time()

# capture a frame
while True:
    ret, frame = cap.read()

    # check if the frame was successfully captured
    if not ret:
        print("Error capturing frame.")
        break

    # display the frame
    cv2.imshow('Video', frame)

    # save the frame as an image file every max_time seconds
    if time.time() - start_time >= max_time:
        cv2.imwrite(f"captured_image_{counter}.png", frame)
        counter += 1
        start_time = time.time()

    # break out of the loop if counter reaches 7
    if counter == 10:
        break

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q') or time.time() - start_time >= 60:
        break

# release the camera and close the window
cap.release()
cv2.destroyAllWindows()
