# Surface Crack Detection and Localization using Edge Impulse

## Overview

Cracks are one of the important criteria used for diagnosing the deterioration of concrete structures. Typically, a specialist would inspect such structures by checking for cracks visually, sketching the results of the inspection, and then preparing inspection data based on their findings. An inspection method like this is not only very time-consuming and costly but it also cannot accurately detect cracks. In this project, I built a surface crack detection application using Seeed reTerminal (based on Raspberry Pi Compute Module 4) and Raspberry Pi Camera Module 2.

## Why localization?

Why do we want to localize the detection using an image classification model? Can't we use the object detection model? Yes, we can use the object detection model but we would need to add bounding boxes to thousands of samples manually. Existing object detection models may not be a good choice to auto-annotate these cracks since they are trained on definite shape objects. Repurposing the classification model for localizing the detection saves a lot of effort and still would be able to identify the regions of interest.

## How does it work?

The CNN (convolutional neural networks) with GAP (Global Average Pooling) layers that have been trained for a classification task can also be used for object localization. That is, a GAP-CNN not only tells us what object is contained in the image - it also tells us where the object is in the image, and through no additional work on our part! The localization is expressed as a heat map (class activation map) where the color-coding scheme identifies regions that are relatively important for the GAP-CNN to perform the object identification task.

## Hardware Setup

Since I wanted a compact and portable hardware setup, we will be using Seeed reTerminal which comes with an LCD and buttons in a compact form factor. It is powered by a Raspberry Pi 4 Compute module with 4 GB RAM which would be sufficient for this proof-of-concept project. We would need Raspberry Pi Camera V2 and an acrylic mount for it.

![Hardware](/images/hardware.jpeg)

We would need to open the back cover to access the 15-pin FPC camera connector. Please follow the step-by-step instructions here: https://wiki.seeedstudio.com/reTerminal.

![FPC](/images/FPC-label-1.jpeg)

The camera is connected using the FPC ribbon cable and attached to the reTerminal using the mount.

![with Camera](/images/reterminal_with_cam.jpeg)

## Setup Development Environment
The reTerminal comes with 32-bit Raspberry Pi OS but We will be using 64-bit Raspberry Pi OS for better performance. Please follow the instructions here to flash the OS: https://wiki.seeedstudio.com/reTerminal-FAQ.

To install the python packages, execute the commands below.

```
$ sudo pip3 install seeed-python-reterminal
$ sudo apt install -y libhdf5-dev python3-pyqt5 libatlas-base-dev
$ pip3 install opencv-contrib-python==4.5.3.56
$ pip3 install matplotlib
```

