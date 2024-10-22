## Project: Behavioral Cloning

In this project we are going to teach the car to stay on track.

### Collecting Training Data

The simulator has been run on Linux 64 bit, smallest size (640x480) and fastest graphical settings.  After activating recording in training mode we will get saved 3 images every 0.1 second - one center camera image, one left and one right camera image.  Image file location, steering angle, throttle and speed are recorded in driving_log.csv file.  The images have dimensions 160x320 pixels and are stored in color RGB uint8 format.

There are 2 car tracks.  I have collected about 15,000 images on the first track (3 laps) and about 14,000 images on the second track (going to the end in both directions).  I was trying to keep the car in the center and I was slowing down at sharp turns or at reduced road visibility but I did not trained specially for recovery because I wanted to create a scenario closer to a real car experiment.

### Image Preprocessing

The saved image is 320 pixels wide and 160 pixels high.  

![Original Image](./pic/img0.png  "img0 - 160x320")

At importing the image I ignore the lower 20 pixels - because this part contains the hood of the car and I don't want the network to figure out whether I'm using left, right or center camera.  I also ignore the upper 12 pixels, mainly to have in the end a nice 128 pixels height.  These upper twelve pixels are mainly clouds, usually not relevant for road direction.

![Stripped Image](./pic/img.png  "img - 128x320")

A Gaussian blur is applied with kernel size (3,3)

![Blurred Image](./pic/img_blur.png  "img_blur - 128x320")

 Vertical and horizontal size is shrunk to half so the image dimension is becoming 64x160 pixels.  

![Half Size Image](./pic/img_half.png  "img_half - 64x160")

Every  image is flipped around the vertical axis:

![Flipped Image](./pic/img_flip.png  "img_flip - 64x160")

 Both images (before and after flip) are added to the input sequence after converting to float32 and normalizing to the range [-0.5, 0.5]

### Label Preprocessing

The steering_angle in my training data is in range [-0.6, 0,6] with most values around 0 and slight left turns on track1 (closed circuit):

![Track1 steering angle](./pic/tr1_steering_angle.png  "Track1 steering angle")

Left and right turns have almost symmetric distribution on track2:

![Track2 steering angle](./pic/tr2_steering_angle.png  "Track2 steering angle")


Steering angle corresponding to center camera is left unchanged.  Values corresponding to left camera are added a shift value about 0.15 and for the right camera we subtract this value.  These shift values are about 2..3 times more what the model predicted for only left or only right camera images when the training has been done on center camera images only.  When driving we are going to use only the center camera images for prediction and the augmented shift values will instruct the car to go back towards the center of the road.

I decided to make the simulation even more realistic and I'm storing and predicting the speed values as well.  Originally the speed is in range [0, 30].  
On track1 most of the time I was going with maximum speed:

![Track1 speed distribution](./pic/tr1_speed.png  "Track1 speed distribution")

On track2 speed is more evenly distributed with smaller peeks around 15, 20 and 30:

![Track2 speed distribution](./pic/tr2_speed.png  "Track2 speed distribution")

 I set all speed less than 10 to 10 supposing that this speed is safe enough to take the sharpest turns in a realistic scenario.  
The  speed data is normalized to a range [-0.2, 0.2].  This range is  smaller than the steering angle range because speed accuracy is not so critical.

### The Model

There are 4 CNN layers and 3 fully connected layers.  The CNN layers have kernel size (5,5) except the last layer which has kernel size (3,3).  Every layer has batch normalization (to speed up learning).  
I've started initially with 3 CNN layers and 2 fully connected layers.  I've increased the number of layers hoping that I can unify track1 and track2 with and without speed control in one single model weight, but in the end I stayed with different model weights for the two tracks.

The CNN layers got ELU activation (for fast and accurate learning) and max pooling with pool size (2,2).  
I've started with ReLU activation but during neural network investigation I've stumbled upon the [ELU activation function](https://arxiv.org/pdf/1511.07289v1.pdf)  description which was very convincing.  The ELU has a good shape for values near zero, which resulted in a quicker and more accurate learning.

![ELU, ReLU and LReLU](./pic/elu_relu_lrelu.png  "ELU, ReLU and LReLU")

After the CNN layers the 96 "images" of 4x10 "pixels" are flattened out and followed by 3 fully connected layers with 2048, 512 and 128 neurons including a dropout and a sigmoid activation.  The output dimension is 2 - corresponding to the steering angle and to the speed values.


### The Training

The batch size is set to 32 and we are going to use Adam optimizer.  The small batch size gives us a quicker learning and the Adam optimizer is going to adjust the learning rate.
The validation data takes 5% of the training samples.

On track1 convergence happens after 2 epochs while track2 is converging after 9 epochs.  Increasing the number of epochs above these values results in overfitting.  
The huge learning time difference between the tracks is because on track1 we are going on circle, the scenery is repeating and spread in speed and steering angle values is much more reduced then scenery diversity and label values spread on track2.


### Running Drive Predictions

Before steering angle and speed predictions we are doing the same image preprocessing as done for model training.  
The predicted steering angle can be sent directly to the simulator.  The predicted speed is transformed back to the range [10..30]; it is then compared to the car actual speed and throttle 0 or 1 is sent to the simulator according to the comparison result.

**Track1**

Use command 

```python drive.py model.json```

to start autonomous driving in "classical" mode  with full throttle and maximum speed.

Use command

```python drive.py model.json -s```

to start autonomous driving with steering and speed control.


**Track2**


To check autonomous driving on track2 use model2 instead of model, e.g.:

```python drive.py model2.json```

and use -s flag to activate speed control.

```python drive.py model2.json -s```






















