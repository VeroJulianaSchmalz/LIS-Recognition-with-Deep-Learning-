### Import libraries 

import tensorflow as tf 
import h5py
import os
from time import sleep
import cv2
import numpy as np
from tensorflow.keras.models import load_model
                         
model=load_model(model_path/where_stored)

### Find the running average over the background

bg = None

def run_avg(image, aWeight):           #basically subtract the background of the are where the sign is performed
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

# Segmenting the region of hand in the image

def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (_, cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

### Accessing the webcam and defining region of interest for gesture placement

aWeight = 0.5

# get the reference to the webcam
camera = cv2.VideoCapture(0)

# region of interest (ROI) coordinates
top, right, bottom, left = 60, 50, 200, 200   

# initialize num of frames
num_frames = 0

# List of classes and reference images from LIS

lista=os.listdir('Code/Segni LIS')
lista

### Getting frame to make predictions 

# get the current frame
(grabbed, frame) = camera.read()

# flip the frame so that it is not the mirror view
frame = cv2.flip(frame, 1)


# keep looping, until interrupted
while(True):
    # get the current frame
    (grabbed, frame) = camera.read()

    # resize the frame
    #frame = imutils.resize(frame, width=700)

    # flip the frame so that it is not the mirror view
    frame = cv2.flip(frame, 1)

    # clone the frame
    clone = frame.copy()

    # get the height and width of the frame
    (height, width) = frame.shape[:2]

    # get the ROI  --> our small square 
    roi = frame[top:bottom, right:left]

    # convert the roi to GRAYSCALE and blur it

    gray=roi

    k=2
    resized = cv2.resize(roi, (28*k,28*k), interpolation = cv2.INTER_AREA)/255         #resize to pass the model an image of the same size used for training 

    #Model predictions

    pred=model.predict(resized.reshape(-1,28*k,28*k,3))
    abc = 'ABCDEFHIKLMNOPQRTUVWXY'

#       
    index=np.argsort(pred)
#        print(index)

    # Consider 3 last predictions (best, 2nd and 3rd)
    tres=index[-3:][0]
    l3=abc[tres[0]]
    l2=abc[tres[1]]
    l1=abc[tres[2]]


    letra=abc[np.argmax(pred)]
    
    # draw the segmented hand
    cv2.rectangle(clone, (left, top), (right, bottom), (255, 255, 255), 2)
    cv2.putText(clone, letra, (left-90, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  #CHANGED HERE


    cv2.putText(clone, l2, (left-150, top+190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)  #CHANGED HERE
    cv2.putText(clone, l3, (left-10, top+190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
   
    # increment the number of frame
    num_frames += 1

    # display the frame with segmented hand
    cv2.imshow("Video Feed", clone)



## Optional to display reference image 
    keypress2 = cv2.waitKey(1) 
    if keypress2 == ord(" "):
        letrica=lista[np.random.randint(24)]
        letraimagen=cv2.imread('Code/Segni LIS'+ letrica)                #Code/Segni LIS
        cloneletrica = letraimagen.copy()


        # Using cv2.putText() method 
        letraimagen = cv2.putText(letraimagen, str(letrica[0]), (10, 50), cv2.FONT_HERSHEY_COMPLEX , 2, (226,43,138), 2, cv2.LINE_AA) 
        
        #cv2.FONT_HERSHEY_SIMPLEX
        
        # Displaying the image 
        cv2.imshow("Letra", letraimagen) 
    #    sleep(5)         



    # Press "q" on the keyboard to stop looping and capturing your video 

    keypress = cv2.waitKey(1) 
    if keypress == ord("q"):
        break


# free up memory
camera.release()
cv2.destroyAllWindows()


