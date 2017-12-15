# create_training_data.py

import numpy as np
from grabscreen import grab_screen
import cv2
from getkeys import key_check


def keys_to_output(keys):

    # [1,0] for jump [0,1] for hold
    output = [0,0]

    if ' ' in keys:
        output[0] = 1
    else:
        output[1] = 1
    return output


file_name = 'training_data.npy'
training_data = []


def main():

    while(True):

        # 800x600 region from the top right of the screen.
        screen = grab_screen(region=(0,40, 288, 512))

        # make the image black and white to reduce data size (3 times smaller in theory)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        # resize the screen for CNN (smaller  =  faster)
        screen = cv2.resize(screen, (80, 60))

        keys = key_check()
        output = keys_to_output(keys)
        training_data.append([screen,output])

        if len(training_data) % 500 == 0:
            print(len(training_data))
            np.save(file_name,training_data)

        keys = key_check()

main()
