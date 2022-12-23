import numpy as np
import gym
from gym import spaces
import logging
import numpy as np
import random
#import gym
import numpy as np
from collections import deque
#from keras.models import Sequential 
#from keras.layers import Dense
#from keras.optimizers import Adam
import os
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from jtop import jtop
import RPi.GPIO as GPIO
import time


with jtop() as jetson:
    xavier_nx = jetson.stats

    CPU_temperature = xavier_nx['Temp CPU']
    GPU_temperature = xavier_nx['Temp GPU']
    Thermal_temperature = xavier_nx['Temp thermal']
    # print('GPU Temp ' , GPU_temperature)
    
# Pin Definitions
output_pin = 29  # BCM pin 18, BOARD pin 12

def main():
    # Pin Setup:
    GPIO.setmode(GPIO.BOARD)  # BCM pin-numbering scheme from Raspberry Pi
    # set pin as an output pin with optional initial state of HIGH
    GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.HIGH)

    print("Starting demo now! Press CTRL+C to exit")
    print(' GPIO info ', GPIO.JETSON_INFO)
    print(' GPIO info ', GPIO.VERSION)
    curr_value = GPIO.HIGH
    try:
        while True:
            time.sleep(5)
            print('Input value ', GPIO.input(output_pin))
            with jtop() as jetson:
                xavier_nx = jetson.stats
                GPU_temperature = xavier_nx['Temp GPU']
                CPU_temperature = xavier_nx['Temp CPU']
                Thermal_temperature = xavier_nx['Temp thermal']
                print('GPU Temperature: ', GPU_temperature)
                if GPU_temperature > 40:
                    curr_value = GPIO.HIGH
                    print("Outputting {} to pin {}".format(curr_value, output_pin))
                    GPIO.output(output_pin, curr_value)
                    #curr_value ^= GPIO.HIGH
                else:
                    curr_value = GPIO.LOW
                    print("Outputting {} to pin {}".format(curr_value, output_pin))
                    GPIO.output(output_pin, curr_value)
                    #curr_value ^= GPIO.HIGH
            # Toggle the output every second
            #print("Outputting {} to pin {}".format(curr_value, output_pin))
            #GPIO.output(output_pin, curr_value)
            #curr_value ^= GPIO.HIGH
    finally:
        GPIO.cleanup()

if __name__ == '__main__':
    main()
