import argparse
import base64
import json
import random
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
from scipy import misc
import cv2

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    print('current throttle:{:.1f}  steering:{:6.3f}  speed:{:5.2f}'.format(
                                  float(throttle), float(steering_angle), float(speed)), end='   ')
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)[12:140,:,:]
    image_array = cv2.GaussianBlur(image_array,(3,3),0)
    image_array = misc.imresize(image_array, 0.5)
    image_array = image_array.astype(np.float32)
    image_array = image_array/255. - 0.5
#    print('image_array.shape:',image_array.shape)                        # (160, 320, 3)
#    transformed_image_array = image_array.reshape(1, *image_array.shape) # (1, 160, 320, 3)
    transformed_image_array = image_array[None, :, :, :]                  # (1, 160, 320, 3)
#    print('transformed_image_array.shape:',transformed_image_array.shape)

    # This model currently assumes that the features of the model are just the images. Feel free to change this.
#    y_pred = model.predict(transformed_image_array, batch_size=1)
#    print('y_pred =', y_pred)
    steering_angle, my_speed = model.predict(transformed_image_array, batch_size=1)[0]
    my_speed = my_speed * 50 + 20  # re_normalize
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    if speed_control and float(speed) > my_speed:
        throttle = 0
    else:
        throttle = 1

    print('>>>>> send_control steering:{:6.3f}  (my_speed:{:5.2f})  throttle:{:2d}'.format(
                                                                      steering_angle, my_speed, throttle))
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    parser.add_argument('-s', '--speed_control', action='store_true', help='speed control')
    args = parser.parse_args()

    speed_control = args.speed_control
    print('Speed control:', speed_control)
    if not speed_control:
        print('use -s to turn on speed_control\n')

    print('Loading & compiling', args.model)
    with open(args.model, 'r') as jfile:
        model = model_from_json(json.load(jfile))

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

