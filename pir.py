from gpiozero import LED
from gpiozero import MotionSensor
from gpiozero import DistanceSensor
import RPi.GPIO as GPIO
import time



dis = DistanceSensor(echo=24, trigger=18)
pir = MotionSensor(21)

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
while True:
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
    time.sleep(0.25)
    
        
    print(count)

