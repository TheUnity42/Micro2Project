import tensorflow as tf
import cv2
import os
import sys

def main(model_path):
    # handle relative paths
    model_path = os.path.abspath(model_path)

    print("Loading model from: {}".format(model_path))

    # load the model
    model = tf.keras.models.load_model(model_path)

    print("Model loaded.")

    # connect to webcam
    cap = cv2.VideoCapture(0)

    # run in a loop until user presses 'q' or 'esc'
    while True:
        # read the frame
        ret, frame = cap.read()

        if not ret:
            print("Error reading frame.")
            break

        # convert to RGB
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # resize to fit the model
        frame = cv2.resize(gray, (224, 224))

        # add a dimension to the image
        frame = frame[..., None]

        # predict the class
        prediction = model.predict(frame)

        # print the prediction
        print(prediction)

        # show the frame
        cv2.imshow("frame", frame)

        # wait for a key press
        key = cv2.waitKey(1)

        # if the user presses 'q' or 'esc'
        if key == ord('q') or key == 27:
            break


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python HandTrackTest.py <model_path>")
        sys.exit(1)
    main(sys.argv[1])
    sys.exit(0)
