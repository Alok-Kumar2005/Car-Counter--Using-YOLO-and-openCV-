import numpy as np
from PIL.ImageOps import scale
# from scipy.special import result
from sympy.physics.units import current
# from ultralytics import YOLO
# import cv2
# import cvzone
# import math
# from sort import *  # import everything from the sort file
#
# cap = cv2.VideoCapture('car.mp4')
# # cap = cv2.VideoCapture('--path--')
#
# cap.set(3, 1200)
# cap.set(4, 608)
#
# model = YOLO('../YOLO-Weights/yolov8n.pt')
# classNames = ["person","bottle", "bicycle", "car", "motorbike", "aeroplane", "bus",
#               "train", "truck", "boat", "traffic light", "fire hydrant",
#               "stop sign", "parking meter", "bench", "bird", "cat", "dog",
#               "horse", "sheep", "cow", "elephant", "bear", "zebra",
#               "giraffe", "backpack", "umbrella", "handbag", "tie",
#               "suitcase", "frisbee", "skis", "snowboard", "sports ball",
#               "kite", "baseball bat", "baseball glove", "skateboard",
#               "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#               "fork", "knife", "spoon", "bowl", "banana", "apple",
#               "sandwich", "orange", "broccoli", "carrot", "hot dog",
#               "pizza", "donut", "cake", "chair", "sofa", "pottedplant",
#               "bed", "diningtable", "toilet", "tvmonitor", "laptop",
#               "mouse", "remote", "keyboard", "cell phone", "microwave",
#               "oven", "toaster", "sink", "refrigerator", "book", "clock",
#               "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
#
# ## Double hastag line are code to apply mask
# ## Code to apply mask (means using only some region of video not full video)
# mask = cv2.imread('mask2.png') # blur all the part of the image where we are not interested
# mask = cv2.resize(mask , (1200,608))
#
#
# ## Tracking using Sort.py
# tracker = Sort(max_age = 20 , min_hits = 3 , iou_threshold=0.3)
#
#
# # Creating a line to count the vehicle when they cross the line
# limits = [423 , 297 , 673 , 297]  # coordinate of the line on the video (choose according to our requirements )
# totalCount = []
#
# while True:
#     success , img  = cap.read()
#     ## mask code
#     imgRegion = cv2.bitwise_and(img , mask)
#     results = model(imgRegion , stream = True)
#
#     # img = cv2.flip(img  , 1)
#     # results = model(img , stream = True)
#
#     ## tracking using sort code
#     detections = np.empty((0,5))
#
#     # finding boxes
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             ## Bounding box
#             x1 , y1 , x2 , y2 = box.xyxy[0]
#             x1,y1,x2,y2 = int(x1) , int(y1) , int(x2) , int(y2)
#             #cv2.rectangle(img , (x1 , y1) , (x2 , y2) , (245 , 45 , 54) , 3)
#             w,h = x2-x1 , y2-y1
#
#             # print percentage or confidence
#             conf = math.ceil((box.conf[0]*100))/100
#             print(conf)
#             cvzone.putTextRect(img , f'{conf}' , (max(0,x1) , max(35,y1)))
#
#             # Class name prdiction
#             cls = int(box.cls[0])
#             currentClass = classNames[cls]
#
#             # to detect only cars
#             #if currentClass == "car":
#             #   cvzone.putTextRect(img , f'{classNames[cls]} {conf}' ,
#             #                      (max(0,x1) , max(35,y1)) , scale = 0.7 , thickness = 1)
#
#             # to detect car, bus , truck
#             if currentClass == 'car' or currentClass == 'Bus' or currentClass == 'Truck' and conf > 0.3:
#                 #cvzone.putTextRect(img , f'{classNames[cls]} {conf}' ,
#                 #                   (max(0,x1) , max(35,y1)) , scale = 0.7 , thickness = 1)
#                 #cvzone.cornerRect(img , (x1,y1,w,h) , l = 15)
#
#                 # Tracking code using sort.py
#                 currentArray = np.vstack([x1,y1,x2,y2,conf])
#                 detections = np.vstack((detections , currentArray))
#
#     # tracking code using sort.py
#     resultsTracker = tracker.update(detections)
#
#     # line code
#     cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 3)
#
#     #  tracking code using sort.py
#     for result in resultsTracker:
#         x1, y1, x2, y2, id = result
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#         print(result)
#         w, h = x2 - x1, y2 - y1
#         cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
#         cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
#                            scale=2, thickness=3, offset=10)
#
#         # making the circle when they cross the line we count it
#         cx , cy = x1+w//2 , y1+h//2
#         cv2.circle(img , (cx , cy) , 5 , (255 ,0,255 ) , cv2.FILLED)
#
#         if limits[0] < cx < limits[2] and limits[1]-20 < cy < limits[1]+20:  # making +-20 range of line
#             if totalCount.append(id) == 0:     # in the range it count may be multiple times to overcome these problem these is the solution
#                 totalCount.append(id)
#                 cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 3) # changes the color of the line from red to green when any vehicle crosses the line
#
#     # Displaying count
#     cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
#
#     cv2.imshow('Video' , img)
#     ## mask code
#     ##cv2.imshow('Video Region' , imgRegion)
#     cv2.waitKey(1)
#
#
#



import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("car.mp4")
cap.set(3, 1200)
cap.set(4, 608)

model = YOLO("../Yolo-Weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("mask2.png")
mask = cv2.resize(mask , (1200,608))

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [300, 450, 573, 450]
totalCount = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    # to show the car count in the Video (not important)
    imgGraphics = cv2.imread("car_image.jpeg", cv2.IMREAD_UNCHANGED)

    #If imgGraphics has 3 channels, convert to 4 channels
    if imgGraphics.shape[2] == 3:
        imgGraphics = cv2.cvtColor(imgGraphics, cv2.COLOR_BGR2BGRA)

    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
                    or currentClass == "motorbike" and conf > 0.3:
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)

                currentArray = np.array([x1, y1, x2, y2, conf])  # store all the boxes in the array
                detections = np.vstack((detections, currentArray))  # append the boxes in the detection from the currentArray array

    resultsTracker = tracker.update(detections)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=3, rt=1, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=0.5, thickness=1, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    cv2.putText(img,str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)

    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)

