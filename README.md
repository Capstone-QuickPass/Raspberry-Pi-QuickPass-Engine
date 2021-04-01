# QuickPass Engine

## Version 0.1.0 March 2020



After installing necessary libraries:
## For Raspberry Pi:
`python mask_detector.py`

## For any other Computer:
`python mask_detector_computer.py`

## New Improvements:
- Capacity Management integrated:
    - PIR Sensor to detect movement out of facility
    - Distance Sensor to ensure individual has left radius
- New Audio for better interaction with individuals entering/leaving
- Hyperparameter tuning for Mask Detection model for better performance



## Version 0.0.1 December 2020

Using TensorFlow Lite Object Detection Model to detect the existence of a person.
#### Functionalities:
- Draws a Region of Interest (ROI) box around a potential person and displays a likelihood score.
- If the likelihood score > 0.60 (60%), a photo is taken and stored in the base project directory.
#### How To Run:
Make sure numpy, picamera and pillow are installed
Install Tensorflow Lite:

-`curl -O http://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip -d ${DATA_DIR}
rm coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip`

Start Program:

-`python3 detect_picamera.py`
