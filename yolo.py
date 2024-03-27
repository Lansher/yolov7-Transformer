import cv2
import colorsys
import numpy as np

import torch

import torch.nn as nn
from PIL import ImageFont, ImageDraw
from nets.yolo import YoloBody
from utils.utils import cvtColor, resize_image, get_classes, get_anchors, preprocess_input, show_config
from utils.utils_bbox import DecodeBox

# why input 'object' to YOLO class ??
class YOLO(object):
    _defaults = {

        "model_path"      : 'model_data/yolov7_weights.pth', 
        "classes_path"    : 'model_data/coco_classes.txt', 
        "anchors_path"    : 'model_data/yolo_anchors.txt', 
        "anchors_mask"    : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],

        "input_shape"     : [640, 640], 

        "phi"             : 'l', 

        "confidence"      : 0.5,

        "nms_iou"         : 0.3, 
        "letterbox_image" : True, 
        "cuda"            : True  




    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        
    #--------------------------------------#
    #   Initialize YOLO
    #--------------------------------------#    
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value, in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value

        #--------------------------------------#
        #   Get number of classes and anchors
        #--------------------------------------#    
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors     = get_anchors(self.anchors_path)
        # Inputs: self, anchors, num_classes, input_shape, anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]
        self.bbox_util                     = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)

        hsv_tuples  = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        # Call generate function in here 
        self.generate()

        # Print configure
        show_config(**self._defaults)
    #--------------------------------------#
    #   From YoloBody class generate a net 
    #--------------------------------------#
    def generate(self, onnx=False):
        self.net    = YoloBody(self.anchors_mask, self.num_classes, self.phi)
        # Set device using torch 
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    =self.net.fuse().eval()
        print('{} model, and classes have loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    #--------------------------------------#
    #   Detect images
    #  crop : save the sigle object after detect or not
    # count : count every detected object or not
    #--------------------------------------#    
    def detect_image(self, image, crop=False, count=False):

        #------------------------------------------------------#
        #   Get width and hight from input image
        #------------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])

        image       = cvtColor(image)

        #------------------------------------------------------#
        #
        #------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #------------------------------------------------------#
        #   Add dimension for 'batch_size'
        #   h*w*3 -> 3*h*w(transpose)-> 1*3*h*w(add dimension)
        #------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():

            # create a object from 'cuda compute' class
            images = torch.from_numpy(image_data)
            if self.cuda:
                image = images.cuda()
            #------------------------------------------------------#
            #   Input the image into Network
            #------------------------------------------------------#
            outputs = self.net(image)
            # decode_box is a function in DecodeBox
            outputs = self.bbox_util.decode_box(outputs)
            #------------------------------------------------------#
            #   Non Max Suppression
            #------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                            image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.num_iou)
            
            if results[0] is None:
                return image
            # ??
            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf  = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]

        #------------------------------------------------------#
        #   Set font and frame thickness
        #------------------------------------------------------#
        font      = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        #------------------------------------------------------#
        # Draw predict box on image
        #------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top    = max(0, np.floor(top).astype('int32'))
            left   = max(0, np.floor(left).astype('int32'))
            bottom = max(image.size[1], np.floor(bottom).astype('int32'))
            right = max(image.size[0], np.floor(right).astype('int32'))

            label = '{}{:.2f}'.format(predicted_class, score)
            draw  = ImageDraw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'utf-8'), fill=(0, 0, 0), font=font)
            del draw
        return image





        
