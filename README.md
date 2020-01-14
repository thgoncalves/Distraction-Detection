# Distraction Detection

## Abstract

A person's attention to a specific task is not a trivial subject. We are often subject to various types of stimuli coming from all around us such as emails, phone calls, messages. All of those events are disputing for our attention. But some tasks are too critical and require us to keep a higher level of concentration, like Driving. With the intention of reducing car crashes, I'll focus on a Machine Learning system that will detect, anticipate and alert drivers of possible crashes due to a driver being distracted.

## The project

We are online and accessible as never before. We can be reached over phone, email, messages, virtually anywhere and at anytime. On the positive side, we are more efficient and productive, but on the negative side we sometimes lose perspective of what is more important at that exact moment. 

The goal of this project is to build a system capable of detecting if a driver has lost their attention on the road to any other stimuli. I'll focus on simple scenarios first, as the usage of cell phones and holding objects on hand (like cups) for a long period of time. The system will take advantage of Machine Learning and Computer Vision to learn the common types of driver behaviors to predict distraction on real time scenarios. My Machine Learning engine will be using TensorFlow and CNN. I'’l train and test the model on images from my dataset and use the frames from Computer Vision to run real life  detections.

## The source

This project is, in a nutshell, a Categorization Problem. And for that, it depends heavily on a good dataset. Fortunately, I was able to find on Kaggle a great data set provided by **State Farm**. On the link below, you can find their information as well as the dataset of images and labels they provided.

Source: https://www.kaggle.com/c/state-farm-distracted-driver-detection/overview

## The Process

![](/images/Model_sequence.png)


1) Collect images

The base for every prediction model depends heavily on the dataset used to model, train on test. I extracted all images and labels to train my model from Kaggle, published by State Farm and the dataset name is Distracted Driver Detection.

 2) Images
 
I need plenty of Images for my engine to be accurate enough to make relevant predictions and they need to be balanced, and luckly they are, as we can see on the bar chart.

![](/images/Label_distribution.png)

3) Machine Learn Engine

This will be the main processing part of my project. All images and labels will be analysed by the model I am going to build and have their labels as the predicted values. This will be a classification problem.

4) Computer Vision

With a prediction model in place, I'’l use OpenCV (computer Vision) to capture frames on a webcam and have those frames processed by the ML engine to make predictions. The result will be displayed back on the Computer Vision in the form of an alert.


## Assumptions

This project is very challenging if consider all possible distractions a person may go through while driving. To site a few, the driver might be look too long to an accident on the side of the road, or falling asleep, or attending to a child on the back seat. There are plenty of use cases for distractions and for each of them but due to time constraint, I've decided to classify only 10 classes (that are provided on the Dataset). Those classes will restrict my model, making it
possible to build in such a short window of time.

Another assumption I am adding is the position of the camera. Ideally the images would be taken somewhere in the dash in front of the driver, just behind the wheel and facing the driver. However was pretty hard to get a good amount of images with the right positioning so I had to switch to photos taken mostly from the passenger front seat. This assumption changes the way Computer Vision demo will work, but still going to be able to process and give signals back.


## Notebooks

1) Uploading Images to AWS

Image classification can take a lot of machine time to complete the processing. That's why I decided to use AWS services, more specifically SageMaker, to run my model.

To do that, I first needed a code to upload the images on my dataset to S3. The first norebook does exactly that

2) CNN Model

 The notebook MobileNet_CNN_Model_vfinal.ipynb has all the code to create the model.
 
 I decided to use the following:
 
 - Keras (tensorflow backended) as the main library
 - Transfer Learning with MobileNet and Imagenet weights
 - I disabled all 87 layers from learning, to try to be less time consuming
 - 4 additional Deep Learning layers, all preceeded with DropOut layers to reduce overfitting
 - Simple Image augumentation, using only rescaling and zooming
 - flowfromdirectory was my selected method to read images for the model. It is a generator, making it faster to read while setting up the model
 
 
2) Image Ramdomizer

This is the Jupyter Notebook I used to display random images from the Test folder to validate the model as a simple way to check the model and display to people around me.
It has `load_model` and `predict` funciton on it
It is the file Image_randomizer.ipynb
 
3) Live feed and OpenCV script

This is the .py script I was displaying on my main monitor, with a live feed from the webcam.
It uses OpenCV to capture images from Webcam and Keras to predict frames from webcam