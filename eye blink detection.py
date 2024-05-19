import random
import cv2 as cv
import cvzone
import mediapipe.python.solutions.drawing_utils
import numpy as np
from mediapipe.python import *
import mediapipe.python.solutions as solutions
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

# ---------------- create detector face ---------------- #
detector = FaceMeshDetector(maxFaces=1)

# ---------------- plot the graph simulator ---------------- #
plot_y = LivePlot(400,400, [20, 50])

# ---------------- List of the left eye landmarks ---------------- #
id_list = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
ratio_list = []

# ---------------- count eye blink ---------------- #
blink_counters = 0
color = (255,0,255)


cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # ---------------- Face Mesh detector ---------------- #
    frame, face = detector.findFaceMesh(frame, draw=False)  ## darw = False >> for not drwing the lm on the face

    # ---------------- check if lm not empty (face) ---------------- #
    if face:
        faces = face[0]
        for id in id_list:
            cv.circle(frame, faces[id],2, color, thickness=cv.FILLED)

        # ---------------- Calculate horizontal and vertical distance ---------------- #
        left_up = faces[159]
        left_down = faces[23]
        left_left = faces[130]
        left_right = faces[243]

        lenght_Horz, _ = detector.findDistance(left_left, left_right)
        lenght_Vert, _ = detector.findDistance(left_up, left_down)

        cv.line(frame, left_up, left_down, color=(255,255,0))
        cv.line(frame, left_left, left_right, color=(255,255,0))

        # ---------------- getting ratio (used for determine blink) ---------------- #
        ratio = (lenght_Vert / lenght_Horz) * 100
        ratio_list.append(ratio)

        if len(ratio_list) > 4:
            ratio_list.pop(0)

        ratio_avg = sum(ratio_list) / len(ratio_list)

        # ---------------- counting blink eye ---------------- #
        if ratio_avg < 37:
            blink_counters += 1
            color = (0,0,255)
        cvzone.putTextRect(frame, f"blink count {blink_counters}", (100, 100), colorR=color)

        color = (255,0,255)

        # ---------------- plot counted blink in graph ---------------- #
        image_plot = plot_y.update(ratio)       ## update the plot with the value of ratio.

        cv.imshow('plot_graph', image_plot)

    cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'):
        break


cap.release()
cv.destroyWindow()