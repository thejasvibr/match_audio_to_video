#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" Get user inputs on regions where the light source and timestamps are 

TODO:
    > speed up the GUI by making a 'proper' interface without matplotlib. 
      the matplotlib TextBox seems to be slowing things down mostly. 
      (https://github.com/matplotlib/matplotlib/issues/8129) -- might disappear 
      in a Python 3 implementation. 

Created on Wed Aug  7 14:34:22 2019

@author: tbeleyur
"""
import argparse
import sys 
sys.setrecursionlimit(90000)
import cv2 
import easygui as eg
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
import numpy as np 

parser = argparse.ArgumentParser()
parser.add_argument('-v','-video_location',type=str, dest='video_path',
                    default='example_data/example_video.avi')
args = parser.parse_args()

class InteractivePlot():
    def __init__(self,figure_axes):
        self.index = 0
        self.figure_axes = figure_axes
    
    def load_video(self):
        self.video = cv2.VideoCapture(self.video_path)
        self.plot_image()

    def plot_image(self):
        self.video.set(1,self.index)
        success, self.frame = self.video.read()
        imshow_axes = self.figure_axes.imshow(self.frame, interpolation='nearest', 
                                                animated=True)
        plt.xticks([])
        plt.yticks([])
        self.figure_axes.set_title('Frame number: '+str(self.index+1))
        imshow_axes.set_data(self.frame)
        plt.show()
    
    def move_forward(self,event):
        self.index +=1
        self.plot_image()

    def move_backward(self,event):
        self.index -= 1 
        self.plot_image()

    def move_to(self, box_input):
        self.index = int(box_input)-1
        self.plot_image()
    
    def change_current_frame_to(self, user_input):
        self.index = int(user_input)-1
        self.plot_image()
        
    
    def radio_on_click(self, label):
        print(label)
        if label =='inspect video':
            print('inspecting video')
            return(None)
        elif label == 'choose border':
            user_points = plt.ginput(4)
            self.xy_inputs = np.fliplr(np.array(user_points))

            print('4 points obtained...')
            # now extract the number of pixels to be cropped from each side
            rows, cols, channels = self.frame.shape
            self.border = (np.min(self.xy_inputs[:,1]), np.min(self.xy_inputs[:,0]),
              cols-np.max(self.xy_inputs[:,1]), rows-np.max(self.xy_inputs[:,0]))
            print('The border is:', self.border)
        elif label == 'change frame':
            frame_number = eg.textbox('Which frame number to go to? Please enter the number with no following spaces and press OK below.') 
            self.change_current_frame_to(int(frame_number))

plt.figure()
a0 = plt.subplot(111)
ip = InteractivePlot(a0)

axnext = plt.axes([0.81,0.05,0.1,0.075])
axprev = plt.axes([0.65,0.05,0.1,0.075])
#framenum = plt.axes([0.45,0.05,0.1,0.075])
rax = plt.axes([0.05, 0.1, 0.15, 0.15])
radio = RadioButtons(rax, ('choose border','change frame'))

#move_to = TextBox(framenum, 'Move To:')
#move_to.on_submit(ip.move_to)
next_button = Button(axnext,'NEXT FRAME')
prev_button = Button(axprev,'PREV FRAME')
next_button.on_clicked(ip.move_forward)
prev_button.on_clicked(ip.move_backward)
radio.on_clicked(ip.radio_on_click)

if __name__ == '__main__':
    # NOTE TO USER : change the ip.video_path if you want to run this from an
    # interactive development environment (IDE) like spyder etc. and want to use 
    #another video -- see line below

    # uncomment for IDE: 
    #ip.video_path = '..//example_data//vlc-record-2019-08-19-16h32m09s-DVRecorder_03_20190704_16.49.45-16.56.42[R][@da37][0].avi'		
    #ip.video_path = "/media/tbeleyur/THEJASVI_DATA_BACKUP_3/fieldwork_2018_002/horseshoe_bat/video/Horseshoe_bat_2018-08/2018-08-18/cam02/OrlovaChukaDome_02_20180819_02.00.00-03.00.00[R][@3513][0].avi"
    ip.video_path = args.video_path 
    ip.load_video()
