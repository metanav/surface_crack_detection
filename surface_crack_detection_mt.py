#!/usr/bin/python3

import os
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

def get_weights(interpreter):
    tensor_details = interpreter.get_tensor_details()
    for d in tensor_details:
        if d['name'] == 'model/prediction/MatMul':
            index = d['index']
    tensor = interpreter.tensor(index)()
    weights = np.copy(tensor)
    return weights

signal.signal(signal.SIGINT, sigint_handler)

def capture(queueIn):
    global terminate
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
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (width, height))
                img = img / 255.0
                img = img.astype(np.float32)

                if not queueIn.full():
                    queueIn.put(img)
                    #logging.info(f'Image Captured time elapsed: {time_elapsed*1000:.1f}ms')
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
                time.sleep(0.2)
                continue

            img = queueIn.get()
            input_data = np.expand_dims(img, axis=0).astype(input_details[0]["dtype"])
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            output_0_tensor = interpreter.tensor(output_details[0]['index'])
            output_1_tensor = interpreter.tensor(output_details[1]['index'])

            pred_class = np.argmax(np.squeeze(output_1_tensor()))
            pred_score = np.squeeze(output_1_tensor())[pred_class]
            dp_out = None

            if pred_class == 1 and show_heatmap is True :
                class_weights = weights[pred_class]
                dp_out = np.dot(
                    np.squeeze(output_0_tensor()).reshape((height*width, 1280)), 
                    class_weights
                ).reshape(height, width)

            queueOut.put((img, pred_class, pred_score, dp_out))
        except Exception as inst:
            logging.error("Exception", inst)
            logging.error(traceback.format_exc())
            break
        logging.info('Inferencing time: {:.3f}ms'.format((time.time() - start_time) * 1000))

def display(queueOut):
    global show_heatmap
    global terminate

    dimension = (960, 720)
    ei_logo = cv2.imread('/home/pi/ei_tflite/ei_logo.jpg')
    ei_logo = cv2.cvtColor(ei_logo, cv2.COLOR_BGR2RGB)
    ei_logo = ei_logo / 255.0
    ei_logo = ei_logo.astype(np.float32)
    ei_logo = cv2.copyMakeBorder(ei_logo, 0, dimension[1] - ei_logo.shape[0], 70, 70, cv2.BORDER_CONSTANT, None, (255, 255, 255))  
    
    fps_counter = avg_fps_counter(30)

    while True:
        if queueOut.empty():
            time.sleep(0.2)
            continue

        img, pred_class, pred_score, dp_out = queueOut.get()

        if pred_class == 1:
            label = 'Crack'
            color = (0, 0, 255)

            if show_heatmap and dp_out is not None:
                colormap = plt.get_cmap('jet')
                heatmap  = (colormap(dp_out) * 2**16).astype(np.float32)[:,:,:3]
                img = cv2.addWeighted(img, 1.0, heatmap, 0.1, 0)
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

        final_img = cv2.putText(final_img, f'FPS:{fps}', (980, 360), font, 2, (0, 0, 0), 3, cv2.LINE_AA)
        final_img = cv2.putText(final_img, f'HM:{"On" if show_heatmap else "Off"}', (980, 440), font, 2, (0, 0, 0), 3, cv2.LINE_AA)

        window_name = "Edge Impulse Inferencing"
        cv2.imshow(window_name, final_img)


        key = cv2.waitKey(1)  
        if key == ord('a'):
            show_heatmap  = not show_heatmap
            logging.info(f"Heatmap: {show_heatmap}")

        if key == ord('f'):
            terminate = True
            logging.info("Display Terminate")
            break
        

if __name__ == '__main__':
    log_fmt = "%(asctime)s: %(message)s"
    logging.basicConfig(format=log_fmt, level=logging.ERROR, datefmt="%H:%M:%S")

    model_file = '/home/pi/ei_tflite/model_160_160_f32.lite'
    interpreter = Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    #logging.debug(input_details)
    output_details = interpreter.get_output_details()
    weights = get_weights(interpreter)

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width  = input_details[0]['shape'][2]
    queueIn  = queue.Queue(maxsize=1)
    queueOut  = queue.Queue()
    show_heatmap = False
    terminate = False

    t1 = threading.Thread(target=capture, args=(queueIn,), daemon=True)
    t2 = threading.Thread(target=inferencing, args=(interpreter, queueIn, queueOut), daemon=True)
    t3 = threading.Thread(target=display, args=(queueOut,), daemon=True)

    t1.start()
    logging.info("Thread start: 2")
    t2.start()
    logging.info("Thread start: 3")
    t3.start()
    logging.info("Thread start: 4")

    t1.join()
    t2.join()
    t3.join()

