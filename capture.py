#####
# Classes Predictor on Live feed from WebCam
# -----------------
# This Python Script was using to run a webcam windows in terminal during the presetation of Demo Day.
#
# I use OpenCV as the Library taht will manage the webcam window and Keras to manage the Model
# The outcome for this code is:
#    1) Display a windows with a live feed from the camera
#    2) Make a prediction based on the live feed every 20 seconds
#    3) Display the predicted value and class on the Webcam window

###########################################################
# PREPARING MODEL (LOADING WEIGTHS AND INSTANTIATE MODEL) #
###########################################################

# Importin Libraries
from keras.models import load_model
from keras.preprocessing import image
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Defining classes name in the form of a Dictionary
labels_string = {'c0':'Safe Driving', 'c1':'Text Rigth',
                 'c2':'Phone Rigth', 'c3':'Text Left',
                 'c4':'Phone Left', 'c5':'Adjusting Radio',
                 'c6':'Drinking', 'c7':'Reaching Behind',
                 'c8':'Hair or Makeup', 'c9':'Talking to Passenger'}

# Load Model based on the Weights calculated by Keras
model = load_model('saved_models/20191207-0450_epochs.h5')


def plot_and_expand_image(path, img_width, img_height):
    '''
    - This function is a simple way to Plot any image of any size using Matplotlib
    -----
    INPUT
    -----
    path (String) -> Path to the image that will be displayed.
                     The path must contain the file name an extenion as well
    img_width (Int) -> Integer with the size of the image
    img_height (Int) -> Integer with the size of the image
    ------
    OUTPUT
    ------
    A NP Array with expanded dimension that will be passe to the model
    '''
    #Load Image
    img = image.load_img(path, target_size=(int(img_width/2), int(img_height/2),3))
    # Convert to Array
    x = image.img_to_array(img)
    # Rescale it (just like it was made on the Augmentation part of the CNN)
    x = x/255.
    # Using Matplotlib mpimg to map the array to a plt object
    img = mpimg.imread(path)
    # Return Function
    return np.expand_dims(x, axis=0)


def predict_image(frame, img_width, img_height):
    '''
    - Predict_image is a function that receives a single image from a local folder and runs a prediction
    on it based on the imported model
    -----
    INPUT
    -----
    frame (String) -> Frame THE PATH to a single jpg image located on the folder. I am calling this input frame because
    later this function will be used to predict frames extracted from a webcam.
    img_width (Int) -> Integer with the size of the image
    img_height (Int) -> Integer with the size of the image
    ------
    OUTPUT
    ------
    class_df (DataFrame) -> This is a DataFrame object containing 11 columns. 
        Column 'image'-> is the local address for the image used to make a prediction
        Columns 'c0','c1','c2','c3','c4','c5','c6','c7','c8','c9' -> are the classes used to make the model
            Classes labels: 'c0':'Safe Driving'
                            'c1':'Text Rigth' 
                            'c2':'Phone Rigth'
                            'c3':'Text Left'
                            'c4':'Phone Left'
                            'c5':'Adjusting Radio'
                            'c6':'Drinking'
                            'c7':'Reaching Behind'
                            'c8':'Hair or Makeup'
                            'c9':'Talking to Passenger'
    '''
    # Create a clean instance of class_df for everytime the function is called
    class_df = pd.DataFrame(columns=['image','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9'])
    # Usinge Keras to load image from the provided path
    x = plot_and_expand_image(frame,img_width, img_height)
    # Make prediction
    y_pred = model.predict(x, batch_size=10)
    # Save prediction to Data Frame
    # First save classes probabilites based on prediction
    class_df[['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']] = pd.DataFrame(y_pred[0]).T
    # Finally save Image path
    class_df['image'] = frame
    #Return DataFrame
    return class_df

################################################################
# PREPARING WEBCAM (LOADING VIDEO AND PASSING FRAMES TO MODEL) #
################################################################

import cv2
# Instantiating webcam
cap = cv2.VideoCapture(0)

i = 0
# Prepare an infinite loop that will keep the webcam working
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Create a clean instance of pred DataFrame that will store results of Predictions
    pred = pd.DataFrame()
    # Point to folder with last saved frame
    path = 'frames/frame.jpg'
    # Defining variables for width and height
    img_width, img_height = 640, 480
    # saving frame to harddrive
    cv2.imwrite(path, frame)
    
    # The Code inside the IF runs once every 10 secons or so
    # This is used to:
    #.  1) Make prediction only every that period of time
    #.  2) write the last prediction on the webcam window
    # I decided to use this approach and not a live stream of data for two reasond:
    #.  1) To be easier on my computer. Making predicitons everytime was too demand on my notebook
    #.  2) To have the text on the screen for long enought so people could be able to read it :)
    
    # For every new cycle
    if i == 0:
        #call prediction function
        predicted_df = predict_image(path, img_width, img_height)
        # Displaying information on screen
        # Removing Image column
        pred = predicted_df.iloc[:,1:]
        # Changing column labels to be Human Readable
        pred.columns = labels_string.values()
        # Identifying class with biggest weight
        maxValue = pred.T[1:].reset_index()[0].idxmax()+1
        
        # create a variable with the Predicted Class label
        pred_label = pred.columns[maxValue]
        # create a variable with the Predicted Class Percentage
        pred_percent = round(pred.iloc[0,maxValue]*100,2)
        # Print on TERMINAL window
        print('------------------------------------------------')
        print(f'Action Detected: {pred_label}')
        print(f'Probability: {pred_percent}%')
        print('------------------------------------------------')
        
        
        # Create a variable with the message for the Webcam Window
        message = f'Action Detected: {pred_label} \nProbability: {pred_percent}%'

    # Now that we have the predicitons and the On Screen Message, we need 
    # to print it to the widow on every single frame, otherwise, everything disapears.
    # There is an order where elements are created. We need to creat bottom elements first and then Top elements
    
    # Creating Rectangle below text
    # Rectangles placement are as follows    
    
    # x1,y1 ------
    # |          |
    # |          |
    # |          |
    # --------x2,y2
    
    # First, create anchors on the top of the element
    x, y = 50,30 # These are x1,y1. Those will also be used to anchor texts later
    frame = cv2.rectangle(frame, # frame information
                          (x, y-30), #rectangle anchor. It has to be displace on Y to cover the whole image
                          (x*20, y*2), # These are x2,y2
                          (0, 255, 0), # rectangle line color
                          cv2.FILLED) # rectangle fill color (green)
    
    # Defining on screen text font
    font = cv2.FONT_HERSHEY_DUPLEX
    # Defining on screen text scale
    fontScale              = 1
    # Defining on screen text color
    fontColor              = (0,0,0)
    # Defining on screen text line type
    lineType               = 2

    # Second, we can add the text over the rectangle
    # In this case, the text has 2 lines and they are spaced 30 units from each other
    dy = 30 # Defining default distance between lines
    for j, line in enumerate(message.split('\n')): # for loop to break the text on every new line (in case I wanted to use more lines)
        y = y + j*dy # Y anchor placement
        frame = cv2.putText(frame, # frame (by this step, this is a composition of the image + rectangle)
                            line, # Anchor position
                            (50, y ), # width and height position
                            font, # font
                            fontScale, # soze
                            fontColor) # color

    cv2.imshow('frame',frame) # dislay frame ( a combination of image + rectangel + text)
    
    # Clock code
    # Every 50 cycles, reset everything
    if i > 50:
        i = -1
    
    # Scape character to break loop by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # display number on screen to show that code is running and not broken
    print(f'{i}          ', end="\r", flush=True)
    i += 1
    
    # END OF WHILE

# If while loop is broken, release webcam
cap.release()
# Destroy all windows created here
cv2.destroyAllWindows()