import cv2
from object_detection import *
import numpy as np

parameters = cv2.aruco.DetectorParameters_create()
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)

detector = ObjectDetection()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()

    corners, _, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    if corners:
        # Aruco Marker çerçeve
        int_corners = np.int0(corners)
        cv2.polylines(frame, int_corners, True, (0, 255, 0), 5)

        # Aruco çevresi
        aruco_perimeter = cv2.arcLength(corners[0], True)
        pixel_cm_ratio = aruco_perimeter / 20

        contours = detector.detect_objects(frame)

        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect

            object_width = w / pixel_cm_ratio
            object_height = h / pixel_cm_ratio

            point = cv2.boxPoints(rect)
            point = np.int0(point)

            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.polylines(frame, [point], True, (255, 0, 0), 2)
            cv2.putText(frame, "X {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
            cv2.putText(frame, "Y {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)



    cv2.imshow("Ölçüm", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()