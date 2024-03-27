import os
import cv2
import random
import numpy as np
import torch
from torch.hub import load_state_dict_from_url
import PIL as Image

#---------------------------------------------------#
#   Set seed for dataloader
#---------------------------------------------------#

def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def resize_image(image, size, letterbox_image, mode='PIL'):
    if mode == 'PIL':
        iw, ih  = image.size
        w, h    = size

        if letterbox_image:
            scale   = min(w/iw, h/ih)
            nw      = int(iw*scale)
            nh      = int(ih*scale)

            image   = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', size, (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        else:
            new_image = image.resize((w, h), Image.BICUBIC)
    else:
        image = np.array(image)
        if letterbox_image:
            # 获得现在的shape
            shape       = np.shape(image)[:2]
            # 获得输出的shape
            if isinstance(size, int):
                size    = (size, size)

            # 计算缩放的比例
            r = min(size[0] / shape[0], size[1] / shape[1])

            # 计算缩放后图片的高宽
            new_unpad   = int(round(shape[1] * r)), int(round(shape[0] * r))
            dw, dh      = size[1] - new_unpad[0], size[0] - new_unpad[1]

            # 除以2以padding到两边
            dw          /= 2  
            dh          /= 2
    
            # 对图像进行resize
            if shape[::-1] != new_unpad:  # resize
                image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
            new_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))  # add border
        else:
            new_image = cv2.resize(image, (w, h))

    return new_image


#---------------------------------------------------#
#   Get class name
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        # read 'voc_classes.txt' in lines
        class_names = f.readlines()
    # Use for loop to save class names in a list
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

#---------------------------------------------------#
#   Get anchors 
#---------------------------------------------------#
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)


def preprocess_input(image):
    image /= 255.0
    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def download_weights(phi, model_dir='./model_data'):
    download_urls = {
        'l' : '', 
        'x' : ''
    }
    url = download_urls[phi]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)
#---------------------------------------------------#
#   Get seed
#---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    # Sets the seed for generating random numbers
    torch.manual_seed(seed)
    # Sets seed for current GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ??
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s| %40s|' % ('keys', 'values'))
    for key, value in kwargs.items():
        print('|%25s| %40s|' % (str(key), str(value)))
    print('-' * 70)