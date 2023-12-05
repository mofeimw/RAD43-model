import cv2 as cv
from ultralytics import YOLO

capture = cv.VideoCapture(0) # capture thru webcam
model = YOLO('weights/best.pt') # load model

while True:
    # read frame and run model
    _, imageFrame = capture.read()
    detections = model(imageFrame)[0]

    # loop through detections
    for data in detections.boxes.data.tolist():
        print(data)
        # discard if confidence under 25%
        confidence = data[4]
        if float(confidence) < 0.25:
            continue

        # classify as walk or stop signal
        signal = data[5]
        if float(signal) == 1.0:
            color = (0, 255, 0) # walk => green
        else:
            color = (0, 0, 255) # stop => red

        # get bounding box coords and draw box
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        imageFrame = cv.rectangle(imageFrame, (xmin, ymin), (xmax, ymax), color, 6)

    # render frame
    cv.imshow('stride', imageFrame)
    cv.waitKey(1)
