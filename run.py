from detectron2.utils.logger import setup_logger
import os
import numpy as np
import cv2
import random
import argparse
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

setup_logger()
'''
parse parameters from terminal
'''
parser = argparse.ArgumentParser()
parser.add_argument("--score_threshold", type=float, default=0.5, help="threshold of confidence score")
parser.add_argument("--mode", type=int, default=2,
                    help="1 means single domain, 2 means the segmentation with adaptation")
parser.add_argument("--input_A", type=str, default='./adaption', help="root path of the dataset after adapted")
parser.add_argument("--input_T", type=str, default='./target', help="root path of the target dataset")
parser.add_argument("--output", type=str, default='./result', help="root path of the segmentation result")
parser.add_argument("--rect_th", type=int, default=1, help="the thickness of bounding box")
parser.add_argument("--text_th", type=int, default=1, help="the thickness of font")
parser.add_argument("--text_size", type=int, default=1, help="the size of font")
parser.add_argument("--show_info", type=bool, default=False, help="whether to show text information on image")
parser.add_argument("--aug_exp", type=bool, default=False, help="whether the augment experiment")
option = parser.parse_args()
print(option)

'''
settings of parameters
'''
rect_th = option.rect_th
text_size = option.text_size
text_th = option.text_th
show_info = option.show_info
mode = option.mode  # 1 means single domain, 2 means the segmentation with adaptation
output_dir = ['mog-aug', 'fog-aug'] if option.aug_exp else ['mog', 'thin', 'thick']
score_threshold = option.score_threshold

'''
load and configuration for instance segmentation model
'''
cfg = get_cfg()
# get configuration of pre-trained model
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
# weights of the model will be downloaded if it is not existing
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
# setup the model for segmentation
predictor = DefaultPredictor(cfg)

'''
color map and class set
'''
colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255],
           [80, 70, 180], [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
# classes of coco objects
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'monitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

'''
auxiliary functions:
@random_colour_masks->get the coloured masks by the size of the input iamge
'''


def random_colour_masks(image):
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0, 10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

  
def author():
    print('Author@Matthew LC Zheng\n'
        'Organization@UESTC\n'
        'Project@Bachelor dissertation: Domain adaptation for isntance segmentation\n'
        'Repository@https://github.com/matthew-lc-zheng/adaptive-seg\n'
        'License@Apache-2.0')
    

'''
@img_r: iamges in target domain
@img_ad: images after adomain adaptation
@require: keep filename of all images aligned with the same length of 10 e.g: 00123_fake.jpg
@instance_segmentation_api->API for single segmentation
@isntance_segmentation_ad_api->API for the segmentation across doamin
'''


def instance_segmentation_ad_api(path_in_ad, path_in_r, path_out, rect_th=1, text_size=3, text_th=1, show_info=True):
    filenames_ad = []
    filenames_r = []
    for _, _, filenames in os.walk(path_in_ad):
        for filename in filenames:
            filenames_ad.append(filename)
            prefix = filename[:6]
            suffix = filename[11:]
            filenames_r.append('%sreal.%s' % (prefix, suffix))

    for j in range(len(filenames_ad)):
        path_img_ad = os.path.join(path_in_ad, filenames_ad[j])
        img_ad = cv2.imread(path_img_ad)
        outputs = predictor(img_ad)
        masks = outputs['instances'].pred_masks.cpu().numpy()
        boxes = outputs['instances'].pred_boxes.tensor.cpu().detach().numpy()
        boxes = [[(i[0], i[1]), (i[2], i[3])] for i in boxes]
        pred_cls = outputs['instances'].pred_classes.cpu().numpy()
        pred_score = outputs['instances'].scores.cpu().numpy()
        path_img_r = os.path.join(path_in_r, filenames_r[j])
        img_r = cv2.imread(path_img_r)
        for i in range(len(masks)):
            rgb_mask = random_colour_masks(masks[i])
            img_r = cv2.addWeighted(img_r, 1, rgb_mask, 0.5, 0)
            (b, g, r) = colours[random.randrange(0, 10)]
            cv2.rectangle(img_r, boxes[i][0], boxes[i][1], color=(b, g, r), thickness=rect_th)
            if show_info:
                cv2.putText(img_r, COCO_INSTANCE_CATEGORY_NAMES[pred_cls[i]] + ":%.2f%%" % (pred_score[i] * 100),
                            boxes[i][0], cv2.FONT_HERSHEY_PLAIN, text_size, (0, 0, 255), thickness=text_th)
        path_save = os.path.join(path_out, filenames_r[j])
        cv2.imwrite(path_save, img_r)


def instance_segmentation_api(path_in, path_out, rect_th=1, text_size=3, text_th=1, show_info=True):
    for _, _, filenames in os.walk(path_in):
        for filename in filenames:
            path_img = os.path.join(path_in, filename)
            img = cv2.imread(path_img)
            outputs = predictor(img)
            masks = outputs['instances'].pred_masks.cpu().numpy()
            boxes = outputs['instances'].pred_boxes.tensor.cpu().detach().numpy()
            boxes = [[(i[0], i[1]), (i[2], i[3])] for i in boxes]
            pred_cls = outputs['instances'].pred_classes.cpu().numpy()
            pred_score = outputs['instances'].scores.cpu().numpy()
            for i in range(len(masks)):
                rgb_mask = random_colour_masks(masks[i])
                img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
                (b, g, r) = colours[random.randrange(0, 10)]
                cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(b, g, r), thickness=rect_th)
                if show_info:
                    cv2.putText(img, COCO_INSTANCE_CATEGORY_NAMES[pred_cls[i]] + ":%.2f%%" % (pred_score[i] * 100),
                                boxes[i][0], cv2.FONT_HERSHEY_PLAIN, text_size, (0, 0, 255), thickness=text_th)
            path_save = os.path.join(path_out, filename)
            cv2.imwrite(path_save, img)


if __name__ == '__main__':
    author()
    for T in output_dir:
        if mode == 1:
            path_in = '%s/%s' % (option.input_T, T)
            path_out = '%s/%s' % (option.output, T)
            instance_segmentation_api(path_in, path_out, rect_th, text_size, text_th, show_info)
        else:
            path_in_r = '%s/%s' % (option.input_T, T)
            path_in_ad = '%s/%s' % (option.input_A, T)
            path_out = '%s/%s' % (option.output, T)
            instance_segmentation_ad_api(path_in_ad, path_in_r, path_out, rect_th, text_size, text_th, show_info)
    
