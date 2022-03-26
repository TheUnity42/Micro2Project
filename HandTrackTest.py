from ctypes.wintypes import RGB
import tensorflow as tf
import numpy as np
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
    cap = cv2.VideoCapture(1)

    # run in a loop until user presses 'q' or 'esc'
    while True:
        # read the frame
        ret, frame = cap.read()

        if not ret:
            print("Error reading frame.")
            break

        # mirror the frame
        frame = cv2.flip(frame, 1)

        # convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # resize to fit the model
        crop_frame = cv2.resize(rgb, (224, 224))
        crop_frame = tf.convert_to_tensor(crop_frame, dtype=tf.float32)

        # predict the class
        prediction = model.predict(crop_frame[None, :, :, :])[0]

        # print the prediction
        print(prediction)

        # calculate the x and y scale factors
        x_scale = frame.shape[1] / 224
        y_scale = frame.shape[0] / 224

        # scale the prediction
        prediction[0] = prediction[0] * y_scale
        prediction[1] = prediction[1] * x_scale
        prediction[2] = prediction[2] * y_scale
        prediction[3] = prediction[3] * x_scale

        # cast to int
        prediction = prediction.astype(np.int32)

        # draw the bounding box
        cv2.rectangle(frame, (prediction[1], prediction[0]), (prediction[3], prediction[2]), (0, 255, 0), 2)

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
