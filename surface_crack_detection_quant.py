#!/usr/bin/python3

import sys
import signal
import time
import cv2
import numpy as np
import traceback
import threading
import logging
import queue
import collections
import matplotlib.pyplot as plt
from matplotlib import cm
from tflite_runtime.interpreter import Interpreter

def avg_fps_counter(window_size):
    window = collections.deque(maxlen=window_size)
    prev = time.monotonic()
    yield 0.0  # First fps value.

    while True:
        curr = time.monotonic()
        window.append(curr - prev)
        prev = curr
        yield len(window) / sum(window)

def sigint_handler(sig, frame):
    logging.info('Interrupted')
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

def capture(queueIn):
    global terminate
    global zoom
    videoCapture = cv2.VideoCapture(0)

    if not videoCapture.isOpened():
        logging.error("Cannot open camera")
        sys.exit(-1)

    while True:
        if terminate:
            logging.info("Capture terminate")
            break

        prev = time.time()
        try:
            success, frame = videoCapture.read()
            if success:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if zoom:
                    w, h = 320, 320
                    x = (img.shape[1] - w) / 2
                    y = (img.shape[0] - h)/ 2
                    img = img[int(y):int(y+h), int(x):int(x+w)]

                img = cv2.resize(img, (width, height))
                img = img / 255.0
                img = img.astype(np.float32)
                img_scaled = (img / input_scale) + input_zero_point
                input_data = np.expand_dims(img_scaled, axis=0).astype(input_details[0]["dtype"])

                if not queueIn.full():
                    queueIn.put((img, input_data))
                    logging.info('Image Captured')
            else:
                raise RuntimeError('Failed to get frame!')
        except Exception as inst:
            logging.error("Exception", inst)
            logging.error(traceback.format_exc())
            videoCapture.release()
            break

def inferencing(interpreter, queueIn, queueOut):
    global terminate
    global show_heatmap

    while True:
        if terminate:
            logging.info("Inferencing terminate")
            break
        start_time = time.time()
        try:
            if queueIn.empty():
                time.sleep(0.01)
                continue

            img, input_data = queueIn.get()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            output_0_tensor = interpreter.tensor(output_details[0]['index'])
            output_1_tensor = interpreter.tensor(output_details[1]['index'])

            output_1 = output_1_scale * ((output_1_tensor()).astype(np.float32) - output_1_zero_point)

            pred_class = np.argmax(np.squeeze(output_1))
            pred_score = np.squeeze(output_1)[pred_class]

            dp_out = None

            if pred_class == 1 and show_heatmap is True :
                dp_out = output_0_scale * (np.squeeze(output_0_tensor())[pred_class].astype(np.float32) - output_0_zero_point)

            if not queueOut.full():
                queueOut.put((img, pred_class, pred_score, dp_out))
        except Exception as inst:
            logging.error("Exception", inst)
            logging.error(traceback.format_exc())
            break
        
        logging.info('Inferencing time: {:.3f}ms'.format((time.time() - start_time) * 1000))

def display(queueOut):
    global show_heatmap
    global zoom
    global terminate

    dimension = (960, 720)
    ei_logo = cv2.imread('/home/pi/surface_crack_detection/ei_logo.jpg')
    ei_logo = cv2.cvtColor(ei_logo, cv2.COLOR_BGR2RGB)
    ei_logo = ei_logo / 255.0
    ei_logo = ei_logo.astype(np.float32)
    ei_logo = cv2.copyMakeBorder(ei_logo, 0, dimension[1] - ei_logo.shape[0], 70, 70, cv2.BORDER_CONSTANT, None, (255, 255, 255))  
    ei_logo = cv2.copyMakeBorder(ei_logo, 0, dimension[1] - ei_logo.shape[0], 70, 70, cv2.BORDER_CONSTANT, None, (255, 255, 255))  
    
    fps_counter = avg_fps_counter(30)

    while True:
        if queueOut.empty():
            time.sleep(0.2)
            continue

        start_time = time.time()
        img, pred_class, pred_score, dp_out = queueOut.get()

        if pred_class == 1:
            label = 'Crack'
            color = (0, 0, 255)

            if show_heatmap and dp_out is not None:
                heatmap = None
                heatmap = cv2.normalize(dp_out, heatmap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                colormap = plt.get_cmap('jet')
                img = cv2.addWeighted(img, 1.0, colormap(heatmap).astype(np.float32)[:,:,:3], 0.4, 0)
        else:
            if pred_class == 0:
                label = 'No Crack'
                color = (0, 0, 0)
            else:
                label = 'Unknown'
                color = (255, 0,  0)
        
        final_img = cv2.resize(img, dimension, interpolation=cv2.INTER_CUBIC) 

        font  = cv2.FONT_HERSHEY_SIMPLEX 
        final_img = np.hstack((final_img, ei_logo))
        final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
        final_img = cv2.putText(final_img, label, (980, 200), font, 2, color, 3, cv2.LINE_AA)
        final_img = cv2.putText(final_img, f'({pred_score*100:0.1f}%)', (980, 280), font, 2, (0, 0, 0), 3, cv2.LINE_AA)

        fps = round(next(fps_counter))

        final_img = cv2.putText(final_img, f'Fps:{fps}', (980, 360), font, 2, (0, 0, 0), 3, cv2.LINE_AA)
        final_img = cv2.putText(final_img, f'Heat:{"On" if show_heatmap else "Off"}', (980, 440), font, 2, (0, 0, 0), 3, cv2.LINE_AA)
        final_img = cv2.putText(final_img, f'Crop:{"On" if zoom else "Off"}', (980, 520), font, 2, (0, 0, 0), 3, cv2.LINE_AA)

        window_name = "Edge Impulse Inferencing"
        cv2.imshow(window_name, final_img)


        key = cv2.waitKey(1)  
        if key == ord('a'):
            show_heatmap  = not show_heatmap
            logging.info(f"Heatmap: {show_heatmap}")

        if key == ord('s'):
            zoom  = not zoom
            logging.info(f"Zoom: {zoom}")

        if key == ord('f'):
            terminate = True
            logging.info("Display Terminate")
            break

        logging.info('Display time: {:.3f}ms'.format((time.time() - start_time) * 1000))
        

if __name__ == '__main__':
    log_fmt = "%(asctime)s: %(message)s"
    logging.basicConfig(format=log_fmt, level=logging.ERROR, datefmt="%H:%M:%S")

    model_file = '/home/pi/surface_crack_detection/model/quantized-model.lite'
    interpreter = Interpreter(model_path=model_file, num_threads=2)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    #logging.debug(input_details)
    output_details = interpreter.get_output_details()

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width  = input_details[0]['shape'][2]
    input_scale, input_zero_point = input_details[0]['quantization']
    output_0_scale, output_0_zero_point = output_details[0]['quantization']
    output_1_scale, output_1_zero_point = output_details[1]['quantization']

    queueIn  = queue.Queue(maxsize=1)
    queueOut  = queue.Queue(maxsize=1)
    show_heatmap = False
    zoom = False
    terminate = False

    t1 = threading.Thread(target=capture, args=(queueIn,), daemon=True)
    t2 = threading.Thread(target=inferencing, args=(interpreter, queueIn, queueOut), daemon=True)
    t3 = threading.Thread(target=display, args=(queueOut,), daemon=True)

    t1.start()
    logging.info("Thread start: 1")
    t2.start()
    logging.info("Thread start: 2")
    t3.start()
    logging.info("Thread start: 3")

    t1.join()
    t2.join()
    t3.join()

