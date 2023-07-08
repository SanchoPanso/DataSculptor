import os
import cv2
import numpy as np
import json
from typing import List, Dict


def read_coco(path: str) -> dict:
    """
    :path: absolute path to json file with coco annotation
    :return: annotation extracted from json file  
    """
    with open(path) as f:
        coco_dict = json.load(f)
        
    coco_categories = coco_dict['categories']
    coco_categories_conformity = {ctg['id']: [i, ctg['name']] for i, ctg in enumerate(coco_categories)}
    
    ctg_ids = list(coco_categories_conformity.keys())
    ctg_ids.sort()  # CHECK
    
    categories = []
    for ctg_id in ctg_ids:
        categories.append(coco_categories_conformity[ctg_id][1])
        
    coco_images = coco_dict['images']
    coco_images_conformity = {img['id']: os.path.splitext(img['file_name'])[0] for img in coco_images}
    labeled_images = {}
    for coco_image in coco_images:
        labeled_images[os.path.splitext(coco_image['file_name'])[0]] = {
            'height': coco_image['height'],
            'width': coco_image['width'],
            'annotations': [],
        }
    
    coco_annotations = coco_dict['annotations']
    for coco_bbox in coco_annotations:
        image_id = coco_bbox['image_id']
        ctg_id = coco_bbox['category_id']
        bbox_coords = coco_bbox['bbox']
        segmentation = coco_bbox['segmentation']
        
        file_name = coco_images_conformity[image_id]
        bbox = {
            'category_id': coco_categories_conformity[ctg_id][0],
            'bbox': bbox_coords,
            'bbox_mode': 'xywh',
            'segmentation': segmentation,
        }
        labeled_images[file_name]['annotations'].append(bbox)
    
    annotation = {
        'categories': categories,
        'images': labeled_images,
    }
    return annotation


def write_yolo_det(annotation: dict, path: str):
    
    os.makedirs(path, exist_ok=True)
    
    for image_name in annotation['images']:
        bboxes = annotation['images'][image_name]['annotations']
        height = annotation['images'][image_name]['height']
        width = annotation['images'][image_name]['width']

        with open(path, 'w') as f:
            for bbox in bboxes:
                cls_id = bbox['category_id']
                if bbox['bbox_mode'] == 'xywhn':
                    x, y, w, h = bbox['bbox']
                elif bbox['bbox_mode'] == 'xywhn':
                    x, y, w, h = xywh2xywhn(bbox['bbox'], (width, height))
                    
                xc = x + w / 2
                yc = y + h / 2
                line = f"{cls_id} {xc} {yc} {w} {h}\n"
                f.write(line)


def write_yolo_iseg(annotation: dict, path: str):
    
    os.makedirs(path, exist_ok=True)
    
    for image_name in annotation['images']:
        bboxes = annotation['images'][image_name]['annotations']
        height = annotation['images'][image_name]['height']
        width = annotation['images'][image_name]['width']
        labels = []
        
        for bbox in bboxes:
            segmentation = bbox['segmentation']
            relative_segmentation = []
            
            if type(segmentation) != list or len(segmentation) == 0 or len(segmentation[0]) <= 4:
                continue
            
            max_seg_contour = find_max_seg_contour(segmentation)
            for i in range(len(max_seg_contour)):
                if i % 2 == 0:
                    x = max_seg_contour[i] / width
                    relative_segmentation.append(x)
                else:
                    y = max_seg_contour[i] / height
                    relative_segmentation.append(y)
            
            label = [bbox['category_id']] + relative_segmentation
            labels.append(label)

        with open(os.path.join(path, image_name + '.txt'), 'w') as f:
            for label in labels:
                f.write(' '.join(list(map(str, label))) + '\n')            


def find_max_seg_contour(segmentation: list) -> list:
    max_idx = 0
    max_square = -1 
    for i, contour in enumerate(segmentation):        
        square = cv2.contourArea(np.array(contour).reshape((-1, 1, 2)))
        if square > max_square:
            max_square = square
            max_idx = i
    return segmentation[max_idx]


def xywh2xywhn(xywh, size):
    x, y, w, h = xywh
    width, height = size
    x /= width
    y /= height
    w /= width
    h /= height
    return (x, y, w, h)


# annot = read_coco(r'C:\Users\HP\Downloads\360_17_0-6\annotations\instances_default.json')
# with open('annot.json', 'w') as f:
#     json.dump(annot, f)
    
