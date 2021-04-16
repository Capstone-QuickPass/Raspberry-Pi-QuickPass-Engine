
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import tensorflow as tf
import time
import requests
from pygame import mixer
from gpiozero import LED
from gpiozero import MotionSensor
from gpiozero import DistanceSensor
from thermal import detect_temp

mixer.init()
light = LED(5)
dis = DistanceSensor(echo=24, trigger=18)
pir = MotionSensor(21)
light.off()
def detect_and_predict_mask(frame, faceNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]


        if confidence > 0.75:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))


            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)


            faces.append(face)
            locs.append((startX, startY, endX, endY))


    if (len(faces) == 1):

                faces = np.array(faces, dtype="float32")
                interpreter = tf.lite.Interpreter(model_path="model.tflite")
                interpreter.allocate_tensors()

                # Get input and output tensors.
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                interpreter.set_tensor(input_details[0]['index'], faces)

                interpreter.invoke()
                preds = interpreter.get_tensor(output_details[0]['index'])
                print(preds)

    return (locs, preds)




print("loading face detector model...")
prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# initialize the video stream and allow the camera sensor to warm up
print("starting video stream...")
vs = VideoStream(src=0).start()
status = False
# player1 = vlc.MediaPlayer("./mask.mp3")
# player2 = vlc.MediaPlayer("./nomask.mp3")


# loop over the frames from the video stream
while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    frames = []
    labels = []
    scores = []
    
    
    maskDetected = False

    (locs, preds) = detect_and_predict_mask(frame, faceNet)
    t = time.localtime()
    print(t)
    current_time = time.strftime("%H:%M:%S", t)
    t  = time.strftime("%m/%d/%Y, %H:%M:%S", t)
    if (len(locs) > 0):
        print("found person")
        mixer.music.load('./capacity_check.mp3')
        time.sleep(2)
        mixer.music.play()
        time.sleep(3)
        fac_info = requests.get('http://quickpass-backend.azurewebsites.net/facility/by/602ea8d423a00b4812b77ee6')
        fac = fac_info.json()
        isCapacitySet = fac['isCapacitySet']
        capacity = fac['capacity']
        currentCapacity = fac['currentCapacity']
        capacity = 1
        currentCapacity = 0
        if capacity <= currentCapacity:
            mixer.music.load('./capacity.mp3')
            time.sleep(1)
            mixer.music.play()
            time.sleep(3.5)
        else:  
            mixer.music.load('./stand.mp3')
            print("standing")
            time.sleep(3)
            mixer.music.play()
            time.sleep(3)
            for i in range(5):
                
                (locs, preds) = detect_and_predict_mask(frame, faceNet)
                for (box, pred) in zip(locs, preds):
                    # unpack the bounding box and predictions
                    (startX, startY, endX, endY) = box
                    (mask, withoutMask) = pred
                    score = max(mask, withoutMask)

                    if (mask > withoutMask):
                        label = "Mask"
                        maskDetected = True
                        color = (0, 255, 0)
                    else:
                        label = "No Mask"
                        color = (0, 0, 255)

                    # include the probability in the label
                    label_new = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                    #develop frame
                    cv2.putText(frame, label_new, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    # show the output frame
                    cv2.imshow("Frame", frame)
                    key = cv2.waitKey(1) & 0xFF

                    # if the `q` key was pressed, break from the loop
                    if key == ord("q"):
                        break

                    frames.append(frame)
                    labels.append(label)
                    scores.append(score)
                time.sleep(0.5)
                
            if maskDetected == True:
                i = labels.index("Mask")
            else:
                i = 0

            if (status==False):
                status = True
                print(maskDetected)
                if maskDetected == True:
                    light.on()
                    mixer.music.load('./mask.mp3')
                    time.sleep(3)
                    mixer.music.play()
                    time.sleep(2) 
                    light.off()
                    mixer.music.load('./temp.mp3')
                    time.sleep(3)
                    mixer.music.play()
                    time.sleep(4)
                else:
                    mixer.music.load('./nomask.mp3')
                    time.sleep(3)
                    mixer.music.play()
                    time.sleep(2)
                    mixer.music.load('./temp.mp3')
                    time.sleep(3)
                    mixer.music.play()
                    time.sleep(4)
                
                temp = detect_temp(4)
                
                print(temp)
                vulnerable = False
                temp_val = 37
                if temp>37:
                    vulnerable = True
                    temp_val = temp
                    mixer.music.load('./temp_alert.mp3')
                    time.sleep(3)
                    mixer.music.play()
                    time.sleep(3)
                elif maskDetected:
                    mixer.music.load('./proceed.mp3')
                    time.sleep(3)
                    mixer.music.play()
                    time.sleep(3)
                x = requests.post('https://quickpass-backend.azurewebsites.net/newPerson', data = {
                    "time": current_time,
                    "score": scores[i],
                    "mask_status": labels[i],
                    "thermalPhoto" : [],
                    "individualPhoto": [],
                
                    "tempValue": temp_val,
                    "datetime": t

                   })
                
            

            maskDetected = False
    elif False:
        fac_info = requests.get('http://quickpass-backend.azurewebsites.net/facility/by/602ea8d423a00b4812b77ee6')
        fac = fac_info.json()
        isCapacitySet = fac['isCapacitySet']
        capacity = fac['capacity']
        currentCapacity = fac['currentCapacity']
        

        def distance():
            GPIO.output(TRIG, True)
            time.sleep(0.00001)
            GPIO.output(TRIG, False)

            while GPIO.input(ECHO)==0:
              pulse_start = time.time()

            while GPIO.input(ECHO)==1:
              pulse_end = time.time()

            pulse_duration = pulse_end - pulse_start

            distance = pulse_duration * 17150

            distance = round(distance, 2)
            
            return (distance)

        count = 0
        while (len(detect_and_predict_mask(imutils.resize(vs.read(), width=400), faceNet)[0])==0):
            time.sleep(2)
            stat = 0
            dis1 = []
            pir.wait_for_motion()
            # time.sleep(3)
            print('Motion Detected')
            for i in range(5):
                currdistance = dis.distance*100
                print('current distance: ', currdistance)
                dis1.append(currdistance)
                if (len(dis1)>1 and dis1[i]>dis1[i-1]):
                    stat+=1
                time.sleep(0.5)

            if (stat>=2):
                count+=1
                print('left the building')
                mixer.music.load('./left.mp3')
                time.sleep(2)
                mixer.music.play()
                time.sleep(3)
                requests.patch('http://quickpass-backend.azurewebsites.net/facility/by/602ea8d423a00b4812b77ee6', data={
                   "currentCapacity":fac["currentCapacity"]-1 
                })
            time.sleep(0.25)
            
                
            print(count)
        


    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    

    if key == ord("q"):
        break


cv2.destroyAllWindows()
vs.stop()
