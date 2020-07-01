import argparse
import dlib
import imutils
from cv2 import cv2
from imutils import face_utils
from scipy.spatial import distance
from imutils.face_utils import FACIAL_LANDMARKS_IDXS
import time
import playsound
from playsound import playsound
from threading import Thread

# function to calculate the eye_aspect_ratio of one eye
def eye_aspect_ration(points):
    # distance between the vertical aspect points
    v1 = distance.euclidean(points[1], points[5])
    v2 = distance.euclidean(points[2], points[4])

    # distance between the horizontal aspect points of the eye
    h = distance.euclidean(points[0], points[3])

    eye_aspect_ratio = (v1 + v2)/(h)
    return eye_aspect_ratio

# function to calculate the average of both eye's eye_aspect_ratio
# and drawing a convexhull around the eye
def average_eye_aspect_ratio_calculator(shape, image):
    # it is used to calculate the set start and end points of
    # left_eye and right_eye's facial_landmark
    (l_start, l_end) = FACIAL_LANDMARKS_IDXS["left_eye"]
    (r_start, r_end) = FACIAL_LANDMARKS_IDXS["right_eye"]

    # slicing of (68*2) matrix to obtain the facial_landmark points
    # of left and right eye
    left_eye_landmark_points = shape[l_start:l_end]
    right_eye_landmark_points = shape[r_start:r_end]

    # calculate the eye_aspect_ratio of each eye (left and right eye)
    left_eye_aspect_ratio = eye_aspect_ration(left_eye_landmark_points)
    right_eye_aspect_ratio = eye_aspect_ration(right_eye_landmark_points)

    # calculate the average aspect ratio of both the eye
    average_eye_aspect_ratio = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2.0

    # draw a convexHull around the eye
    leftEyeHull = cv2.convexHull(left_eye_landmark_points)
    rightEyeHull = cv2.convexHull(right_eye_landmark_points)
    cv2.drawContours(image, [leftEyeHull], -1, (0, 255, 0), 1)
    cv2.drawContours(image, [rightEyeHull], -1, (0, 255, 0), 1)

    return (average_eye_aspect_ratio, image)


def draw_rectangle_over_faces_in_the_image(rects, image):
    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (125, 122, 0), 2)
    return image


def trigger_alarm(audio):
    playsound(audio)


# this function helps in identifying faces in the video stream
# detect the eye_landmark of each face in the image
def drowsiness_detector():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", type=str, default="shape_predictor_68_face_landmarks.dat",
                    help="path to facial landmark predictor")
    ap.add_argument("-t", "--ear-threshold", type=float, default=0.4,
                    help="minimum eye aspect ration threshold to be considered, below this threshold consider it as eye blink")
    ap.add_argument("-c", "--min-frames", type=int, default=15,
                    help="minimum frames which need to considered to confirm that the preson has blinked his eye")
    ap.add_argument("-a", "--audio-file", type=str, default="alarm.mp3",
                    help="path to find the sudio file")
    args = vars(ap.parse_args())

    count = 0
    audio = False
    # capture image from the webcam
    print("[INFO] activating webcam...")
    capture = cv2.VideoCapture(0)

    # initialize dlib face detector (HOG-based) and then create
    # the facial landmark from the dlib predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])

    while True:
        # capture a single frame from the webcam VideoStream
        (success, image) = capture.read()

        # load the input image, resize it, and convert it to grayscale
        image = imutils.resize(image, width=700)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_start_time = time.time()
        # detect faces in the grayscale image
        rects = detector(gray, 1)
        image = draw_rectangle_over_faces_in_the_image(rects, image)

        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region
            shape = predictor(gray, rect)
            # a 68*2 matrix is returned from the below function (shape_to_np)
            shape = face_utils.shape_to_np(shape)

            # here we obtain the average eye_aspect_ratio of both the eye
            (average_eye_aspect_ratio, image) = average_eye_aspect_ratio_calculator(shape, image)

            # to check whether the eye_aspect_ratio is below the threshold limit
            # if yes increase the count
            if average_eye_aspect_ratio < args["ear_threshold"]:
                count += 1
                cv2.putText(image, "Warning!!!", (580, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if count > args["min_frames"]:
                    if not audio:
                        audio = True
                        t = Thread(target= trigger_alarm, args= (args["audio_file"],))
                        # t.daemon = True
                        t.start()
            else:
                count = 0
                audio = False

            # to write eye_aspect_ratio and count of blinks on the image
            cv2.putText(image, "eye_aspect_ratio: {:.2f}".format(average_eye_aspect_ratio), (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # show the output image with the face detections + facial landmarks
        cv2.imshow("Output", image)
        key = cv2.waitKey(1) & 0xFF

        # to quit from the program, Enter 'q'
        if key == ord("q"):
            break

    # once the programs get terminated all the window frames get destroyed
    cv2.destroyAllWindows()


if __name__ == '__main__':
    drowsiness_detector()