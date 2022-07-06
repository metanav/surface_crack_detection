# Surface Crack Detection and Localization using Edge Impulse
### A computer vision system that detects and localizes the surface cracks in the concrete structures for the predictive maintenance

![Cover Image](/images/cover.gif)

## Overview

A 50-year-old bridge collapsed in Pittsburgh (Pennsylvania) on January 28, 2022. There is only one reason why a sturdy structure such as a concrete bridge could suddenly collapse: wear and tear.

![Bridge Collapsed](/images/bridge_collapse.jpeg)

Concrete structures generally start deteriorating after about 40 to 50 years. For this reason, overlooking signs of wear can result in severe accidents, which is why the inspection and repair of concrete structures are crucial for safeguarding our way of life. Cracks are one of the important criteria used for diagnosing the deterioration of concrete structures. Typically, a specialist would inspect such structures by checking for cracks visually, sketching the results of the inspection, and then preparing inspection data based on their findings. An inspection method like this is not only very time-consuming and costly but it also cannot accurately detect cracks. In this project, I built a surface crack detection application using machine learning. A pre-trained image classification model is fine-tuned using the Transfer Learning with the Edge Impulse Studio and deployed to the Seeed reTerminal (based on Raspberry Pi Compute Module 4) which detects surface cracks in real-time and also localizes them.

## Why localization?

Why do we want to localize the detection using an image classification model? Can't we use the object detection model? Yes, we can use the object detection model but we would need to add bounding boxes to thousands of samples manually. Existing object detection models may not be a good choice to auto-annotate these cracks since they are trained on definite shape objects. Repurposing the classification model for localizing the detection saves a lot of effort and still would be able to identify the regions of interest.

## How does it work?

The CNN (convolutional neural networks) with GAP (Global Average Pooling) layers that have been trained for a classification task can also be used for object localization. That is, a GAP-CNN not only tells us what object is contained in the image - it also tells us where the object is in the image, and through no additional work on our part! The localization is expressed as a heat map (class activation map) where the color-coding scheme identifies regions that are relatively important for the GAP-CNN to perform the object identification task.

## Hardware Setup

Since I wanted a compact and portable hardware setup, we will be using Seeed reTerminal which comes with an LCD and buttons in a compact form. It is powered by a Raspberry Pi 4 Compute module with 4 GB RAM which would be sufficient for this proof-of-concept project. We would need Raspberry Pi Camera V2 and an acrylic mount for it.

![Hardware](/images/hardware.jpeg)

We would need to open the back cover of the reTerminal to access the 15-pin FPC camera connector. Please follow the step-by-step instructions here: https://wiki.seeedstudio.com/reTerminal.

![FPC](/images/fpc_label.jpeg)

The camera is connected using the FPC ribbon cable and attached to the reTerminal using the mount.

![with Camera](/images/reterminal_with_cam.jpeg)

## Setup Development Environment
The reTerminal comes with 32-bit Raspberry Pi OS but We will be using 64-bit Raspberry Pi OS for better performance. Please follow the instructions here to flash the 64-bit Raspberry Pi OS: https://wiki.seeedstudio.com/reTerminal-FAQ.

To install the python packages which we will be using in the inferencing code, execute the commands below.

```
$ sudo pip3 install seeed-python-reterminal
$ sudo apt install -y libhdf5-dev python3-pyqt5 libatlas-base-dev
$ pip3 install opencv-contrib-python==4.5.3.56
$ pip3 install matplotlib
```

## Data collection
The datasets were downloaded from the Mendeley Data (Concrete Crack Images for Classification). The dataset contains various concrete surfaces with and without cracks. The data is collected from multiple METU Campus Buildings. The dataset is divided into two negative and positive crack images for image classification. Each class has 20,000 images with a total of 40,000 images with 227 x 227 pixels with RGB channels.

![Datasets](/images/datasets.png)

To differentiate crack and non-crack surface images from the other natural world scenes, 25,000 randomly sampled images for 80 object categories from the COCO-Minitrain, a subset of the COCO train2017 dataset, were downloaded. The data can be accessed from the links below.

- Surface Crack Dataset: https://data.mendeley.com/datasets/5y9wdsg2zt/2
- COCO-Minitrain dataset: https://github.com/giddyyupp/coco-minitrain

## Uploading data to Edge Impulse Studio
We need to create a new project to upload data to Edge Impulse Studio.

![New Project](/images/new_project.png)

The data is uploaded using the Edge Impulse CLI. Please follow the instructions to install the CLI here: https://docs.edgeimpulse.com/docs/cli-installation.

The downloaded images are labeled into 3 classes and saved into the directories with the label name.

- Positive - surface with crack
- Negative - surface without crack
- Unknown - images from the 80 objects

Execute the following commands to upload the images to the Edge Impulse Studio. 
The datasets are automatically split into training and testing datasets.

```
$ edge-impulse-uploader --category split  --label positive positive/*.jpg
$ edge-impulse-uploader --category split  --label negative negative/*.jpg
$ edge-impulse-uploader --category split  --label unknown  unknown/*.jpg
```

We can see the uploaded datasets on the Edge Impulse Studio's Data Acquisition page.

![Data Aquisition](/images/data_aquisition.png)

## Training
Go to the Impulse Design > Create Impulse page, click Add a processing block, and then choose Image, which preprocesses and normalizes image data, and optionally reduces the color depth. Also, on the same page, click Add a learning block, and choose Transfer Learning (Images), which fine-tunes a pre-trained image classification model on the data. We are using a 160x160 image size. Now click on the Save Impulse button.

![Create Impulse](/images/create_impulse.png)

Next, go to the Impulse Design > Image page and set the Color depth parameter as RGB, and click the Save parameters button which redirects to another page where we should click on the Generate Feature button. It usually takes a couple of minutes to complete feature generation.

![Feature Generation](/images/generate_features.png)

We can see the 2D visualization of the generated features in Feature Explorer.

![Feature Explorer](/images/feature_explorer.png)

Now go to the Impulse Design > Transfer Learning page and choose the Neural Network architecture. We are using the MobileNetV2 160x160 1.0 transfer learning model with the pre-trained weight provided by the Edge Impulse Studio.

![Model Selection](/images/choose_model.png)

The pre-trained model outputs the class prediction probabilities. To get the class activation map, we need to modify the model and make it a multi-output model. To customize the model, we need to switch to Keras (expert) mode.

![Switch Expert Mode](/images/switch_expert_mode.png)

We can modify the generated code in the text editor as shown below.

![Editor](/images/model_editor.png)

We will connect the 2nd last layer which is a GAP layer to the Dense layer with 3 neurons ( 3 classes in our case). We will be using this Dense layer weights for generating class activation map later.

```
base_model = tf.keras.applications.MobileNetV2(
    input_shape = INPUT_SHAPE, alpha=1,
    weights = WEIGHTS_PATH
)
last_layer  = base_model.layers[-2].output
dense_layer = Dense(classes)
output_pred = Softmax(name="prediction")(dense_layer(last_layer))
```

For the class activation map, we need to calculate the dot product of the last convolutional block output and the final dense layers' weight. The Keras Dot layer does not broadcast the multiplier vector with the dynamic batch size so we can not use it. But we can take advantage of the Dense layer which internally does the dot product of the kernel weight with the input. There is a side effect in this approach, the Dense layer adds up bias weight to each dot product but since this bias weight is very small and does not changes the final normalized values of the class activation map,

```
conv_layer  = base_model.layers[-4].output
reshape_layer = Reshape((conv_layer.shape[1] * conv_layer.shape[2] , -1))(conv_layer)
dot_output = dense_layer(reshape_layer)
```

We need to resample the dot product output to the same size of the input image (160x160) so that we can overlay the heat map. We use UpSampling2D layer for this purpose.

```
transpose = Permute((2, 1))(dot_output)
reshape_2_layer = Reshape((-1, conv_layer.shape[1] , conv_layer.shape[2]))(transpose)
SIZE = (int(INPUT_SHAPE[1] / conv_layer.shape[2]), 
 int(INPUT_SHAPE[0] / conv_layer.shape[1]))
output_act_map = UpSampling2D(size=SIZE, interpolation="bilinear", data_format="channels_first", name="activation_map")(reshape_2_layer)
model = Model(inputs=base_model.inputs, outputs=[output_pred, output_act_map])
```

Also, we will be training the model from the last two convolutional blocks and freezing all layers before that.

```
TRAINABLE_START_IDX = -12
for layer in model.layers[:TRAINABLE_START_IDX]:
    layer.trainable = False
```

The modified network architecture after the last convolutional block is given below. This is a multi-output model where the first output provides the prediction class probabilities and the second output provides the class activation map.

![Network](/images/model.png)

The full modified training code is as follows.

```
import math
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, UpSampling2D, Permute, Reshape, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy

sys.path.append('./resources/libraries')
import ei_tensorflow.training

WEIGHTS_PATH = './transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160.h5'

INPUT_SHAPE = (160, 160, 3)

base_model = tf.keras.applications.MobileNetV2(
    input_shape = INPUT_SHAPE, alpha=1,
    weights = WEIGHTS_PATH
)

last_layer  = base_model.layers[-2].output
dense_layer = Dense(classes)
output_pred = Softmax(name="prediction")(dense_layer(last_layer))


conv_layer  = base_model.layers[-4].output
reshape_layer = Reshape((conv_layer.shape[1] * conv_layer.shape[2] , -1))(conv_layer)

dot_output = dense_layer(reshape_layer)
transpose = Permute((2, 1))(dot_output)
reshape_2_layer = Reshape((-1, conv_layer.shape[1] , conv_layer.shape[2]))(transpose)

SIZE = (int(INPUT_SHAPE[1] / conv_layer.shape[2]), 
 int(INPUT_SHAPE[0] / conv_layer.shape[1]))

output_act_map = UpSampling2D(size=SIZE, interpolation="bilinear", data_format="channels_first", name="activation_map")(reshape_2_layer)

model = Model(inputs=base_model.inputs, outputs=[output_pred, output_act_map])

TRAINABLE_START_IDX = -12
for layer in model.layers[:TRAINABLE_START_IDX]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
      loss={'prediction': 'categorical_crossentropy', 'activation_map': None},
      metrics={'prediction': ['accuracy'], 'activation_map': [None]})
                
BATCH_SIZE = 32
EPOCHS=5

train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)

validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)

callbacks.append(BatchLoggerCallback(BATCH_SIZE, train_sample_count, epochs=EPOCHS))

model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, verbose=2, callbacks=callbacks)

```


