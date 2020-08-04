from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet_stream as darknet

from flask import Flask, render_template, Response
app = Flask(__name__)
app.config["DEBUG"] = True
total_passed_vehicle = 0 
def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def resizecoordinate(detections, originalwidth, originalheight, newwidth, newheight):
    #newdetections = list(detections)
    for detection in detections:
        detection[2][0] = detection[2][0]/originalwidth*newwidth
        detection[2][1] = detection[2][1]/originalheight*newheight
        detection[2][2] = detection[2][2]/originalwidth*newwidth
        detection[2][3] = detection[2][3]/originalheight*newheight
    #return newdetections

def countclass(detections):
    count_cls_jeff = {}
    for detection in detections:
        nameTag_jeff = detection[0].decode()
        if count_cls_jeff.get(nameTag_jeff) == None:
            count_cls_jeff[nameTag_jeff] = 1
        else:
            count_cls_jeff[nameTag_jeff] += 1
    text = ''
    for key, value in count_cls_jeff.items():
        text += key + ':' + str(value) + ' ' 
    return text

def cvDrawBoxes(detections, img, text, height, width):
    global total_passed_vehicle
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
        cv2.putText(img, text, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], 2)

        # count
        roi = int(height*0.7) # roi line position，這是線的位置
        deviation = 2 # the constant that represents the object counting area
        line_width_start = int(width/2) + 20
        line_width_end = width
        counter = 0
        if x >= line_width_start and x+w <= line_width_end and y + h/2 <= roi + 4 and y + h/2 >= roi - 4 and detection[0].decode() == 'car':
            counter = 1
        # when the vehicle passed over line and counted, make the color of ROI line green
        if counter == 1:                  
            cv2.line(img, (line_width_start, roi), (line_width_end, roi), (0, 0xFF, 0), 5)
        else:
            cv2.line(img, (line_width_start, roi), (line_width_end, roi), (0, 0, 0xFF), 5)
        total_passed_vehicle = total_passed_vehicle + counter
    return img


netMain = None
metaMain = None
altNames = None


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def YOLO():

    global metaMain, netMain, altNames, total_passed_vehicle
    configPath = "./cfg/yolov4.cfg"
    weightPath = "./yolov4.weights"
    metaPath = "./cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("data/hightway.mp4")
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #cap.set(3, 1280)
    #cap.set(4, 720)
    #out = cv2.VideoWriter(
    #    "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
    #    (darknet.network_width(netMain), darknet.network_height(netMain)))
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        height, width, _ = frame_read.shape
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())


        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        resizecoordinate(detections, darknet.network_width(netMain), darknet.network_height(netMain), width, height)
        text = countclass(detections)
        image = cvDrawBoxes(detections, frame_read, text, height, width)
        # insert information text to video frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            image,
            'Counting Vehicles: ' + str(total_passed_vehicle),
            (10, 60),
            font,
            0.8,
            (0, 0xFF, 0xFF),
            2,
            cv2.FONT_HERSHEY_SIMPLEX,
            )  
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.flip(image,1) 
        #print(1/(time.time()-prev_time))
        #cv2.imshow('Demo', image)
        #cv2.waitKey(3)
                     
        
        
        convert = cv2.imencode('.jpg',image)[1].tobytes()
        # _, buffer = cv2.imencode('.jpg',image)
        # convert = buffer.tostring()
        yield(b'--frame\r\n'
              b'Content-Type:image/jpeg\r\n\r\n' + convert +b'\r\n')

    #cap.release()
    #out.release()

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(YOLO(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    #YOLO()
    app.run(host='0.0.0.0', threaded = True, port='5002')
