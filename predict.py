import time
import cv2

import numpy as np

from PIL import Image

from yolo import YOLO

if __name__ == '__main__':
    #----------------------------------------------------------#
    #   set 'mode' to choose mode for testing 
    #   'predict' : predict single image
    #   'heatmap' : visualize heatmap
    #----------------------------------------------------------#

    mode = 'predict'

    crop = False
    count= False

    # video_path = 0
    # vidio_save_path = ''
    # vedio_fps = 25.0

    if mode != 'predict_onnx':
        yolo = YOLO()
    else:
        yolo = YOLO_ONNX()

    if mode == 'predict':
        while True:
            img_path = input('Input image filename: (including "./jpg" or "./png")')
            try:
                image = Image.open(img_path)
            except:
                print('Cannot open image! Please try again')
                continue
            else:
                r_image = yolo.detect_image(image, crop=crop, count=count)
                r_image.show()