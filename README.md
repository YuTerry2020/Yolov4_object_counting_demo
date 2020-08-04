## Yolov4_object_counting_demo
* This project will present how to upload detection and counting car to website.
## Environment
* cuda (choose version by you os and tensorflow version)
* cudnn (choose version by you os and tensorflow version)
* flask
## Python library
* tensorflow-gpu
* openCV
* labelImg
* cmake
* pyyaml

## File explain
* darknet_video_stream.py: It has yolo object detection and flask framework
## How to work
* First, create a environment by Anaconda for yolov4
* Second, download yolov4 darknet
* change Makefile
* python darknet_video_stream.py
## Reference
* Install yolov4
  * https://medium.com/@yanweiliu/training-coco-object-detection-with-yolo-v4-f11bece3feb6
* Download darknet
  * git clone https://github.com/AlexeyAB/darknet.git
