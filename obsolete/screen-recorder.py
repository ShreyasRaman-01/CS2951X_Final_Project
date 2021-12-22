from pynput.keyboard import Key, Listener

import os
import time

import numpy as np
from mss import mss
from PIL import Image

curr_key = 'X'

def on_press(key):
    global curr_key
    last_key = curr_key
    if   str(key) == 'Key.down'  : curr_key = "D"
    elif str(key) == 'Key.up'    : curr_key = "U"
    elif str(key) == 'Key.left'  : curr_key = "L"
    elif str(key) == 'Key.right' : curr_key = "R"
    elif str(key) == '\'k\''     : curr_key = "K"
    elif str(key) == '\'w\''     : curr_key = "W"
    if last_key == 'X' and curr_key != last_key:
        print(" - Recording Started")
    if curr_key == 'W':
        print(" - Recording Stalled: Win Recorded")
    if curr_key == 'K':
        print(" - Recording Stalled: Loss Recorded")

def on_release(key):
    if key == Key.esc:
        return False

def wait(interval, start_time, count):
    '''Function to wait for specific interval'''
    drag = time.time() - start_time - interval * count
    if interval - drag > 0:
        time.sleep(interval - drag)

## Initialize parameters for screen-recording camera
screen_params = {'left': 625, 'top': 110, 'width': 800, 'height': 300}

start_time = time.time()
trial_num = 0               ## Which trial to start at (for trial numbering)
image_size = (320, 120)     ## Size of game screen on export
interval = 0.1              ## Seconds between each 

print("Experiment Started")

## Collect events until released
listener = Listener(on_press=on_press,on_release=on_release)
listener.start()

with mss() as sct:          ## Initialize screenshot utility
    while True:

        while curr_key == 'X':
            ## Wait until an experiment starts
            start_time = time.time()
            frame_num = 0
            continue
        
        curr_dir = f'trials/trial_{trial_num}'
        os.makedirs(f'{curr_dir}/imgs', exist_ok = True)
        with open(f'{curr_dir}/times.txt', 'a') as f: 
            
            ## Make sure to flush keystroke record every 2 seconds
            for _ in range(20):

                ## If not 'X', shoot screen or record latest action
                if curr_key != 'X':
                    shot = sct.grab(screen_params)
                    img = Image.frombytes('RGB', (shot.width, shot.height), shot.rgb)
                    img = img.resize(image_size)
                    img.save(f'{curr_dir}/imgs/screen_{frame_num:06d}.png')
                    f.write(curr_key)
                
                ## If terminal state (Win/Kill), go to next run and set to not record until direction key hit
                if curr_key in ('W', 'K'):
                    trial_num += 1
                    curr_key = 'X'
                    print(f"Starting Experiment {trial_num}: Press Direction Key To Start")
                    break
            
                wait(interval, start_time, frame_num)
                frame_num += 1