# test_model.py

import numpy as np
from grabscreen import grab_screen
import cv2
import time
from directkeys import PressKey,ReleaseKey, SPACE
from alexnet import alexnet
from getkeys import key_check
import tensorflow as tf

import random

WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'flappy-bird-bot-{}-{}-{}-epochs-300K-data.model'.format(LR, 'alexnetv2',EPOCHS)

pause_time = 0.5

def hold():
    #ReleaseKey(SPACE)
    print('Holding, not jumping')


def jump():
    print('jump')
    PressKey(SPACE)
    ReleaseKey(SPACE)
    time.sleep(pause_time)

model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

def main():
    last_time = time.time()

    while(True):
            screen = grab_screen(region=(0,40, 288, 512))
            print('Screen grab took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (160,120))

            prediction = model.predict([screen.reshape(160,120,1)])[0]
            print(prediction[0])
            print(prediction[1])

            #prediction values
            jump_thresh = 0.116
            hold_thresh = 0.88418

            if prediction[1] > jump_thresh:
                hold()
            elif prediction[0] > hold_thresh:
                jump()
            else:
                jump()

main()
