from gpiozero import Servo

servo = Servo(17)
from time import sleep

while True:
    servo.mid()
    print('mid')
    sleep(0.5)
    servo.min()
    print('min')
    sleep(1)
    servo.mid()
    print('mid')
    sleep(0.5)
    servo.max()
    print('max')
    sleep(1)
    