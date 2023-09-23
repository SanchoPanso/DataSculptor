import os
import cv2
import numpy as np
import json
from typing import List, Dict, Union
from easydict import EasyDict


class AnnotatedImage(EasyDict):
    height: int
    width: int


class InstanceImage(AnnotatedImage):
    annotations: list
    

class SemanticImage(AnnotatedImage):
    sem_seg_file_name: str
    

class Annotation(EasyDict):
    """Class that contains annotations for computer vision tasks (detection, instance segmentation)"""
    categories: List[str]
    images: EasyDict[str, AnnotatedImage]
    
    # def __init__(self, categories: List[str] = None, images: EasyDict[str, AnnotatedImage] = None):
    #     """
    #     :param categories: list of categories (or classes), defaults to None
    #     :param images: easydict of labeled images 
    #                    with image names as keys (without file extension), defaults to None
    #     """
    #     super().__init__()
    #     self['categories'] = categories
    #     self['images'] = images


def read_coco(path: Union[str, bytes, os.PathLike]) -> Annotation:
    """ Read COCO annotation format
    
    :path: absolute path to json file with coco annotation
    :return: annotation extracted from json file  
    """
    
    with open(path) as f:
        coco_dict = json.load(f)
    
    # Read list of coco categories which is list dicts with info
    # Then create conformity between ctg id and ctg number with ctg name   
    coco_categories = coco_dict['categories']
    coco_categories_conformity = {}
    for i, ctg in enumerate(coco_categories):
        coco_categories_conformity[ctg['id']] = {'num': i, 'name': ctg['name']}
    
    # Sort category ids
    ctg_ids = list(coco_categories_conformity.keys())
    ctg_ids.sort()  # CHECK
    
    # Create list of category names with order defined by sorted list of ids
    categories = []
    for ctg_id in ctg_ids:
        categories.append(coco_categories_conformity[ctg_id]['name'])
    
    # Create conformity between image id and image name (without ext)
    coco_images = coco_dict['images']
    coco_images_conformity = {img['id']: os.path.splitext(img['file_name'])[0] for img in coco_images}
    
    # Get labeled images
    labeled_images = EasyDict()
    for coco_image in coco_images:
        image_name = os.path.splitext(coco_image['file_name'])[0]
        labeled_images[image_name] = EasyDict({
            'height': coco_image['height'],
            'width': coco_image['width'],
            'annotations': [],
        })
    
    
    coco_annotations = coco_dict['annotations']
    for coco_bbox in coco_annotations:
        image_id = coco_bbox['image_id']
        ctg_id = coco_bbox['category_id']
        bbox_coords = coco_bbox['bbox']
        segmentation = coco_bbox['segmentation']
        
        file_name = coco_images_conformity[image_id]
        bbox = {
            'category_id': coco_categories_conformity[ctg_id]['num'],
            'bbox': bbox_coords,
            'bbox_mode': 'xywh',
            'segmentation': segmentation,
        }
        labeled_images[file_name]['annotations'].append(bbox)
    
    annotation = Annotation({'categories': categories, 'images': labeled_images})
    
    return annotation


def write_coco(annotation: Annotation, path: Union[str, bytes, os.PathLike], image_ext: str = '.jpg'):
    """
    :annotation: annotation to convert
    :path: absolute path for saving coco json  
    :image_ext: file extension, under which images will be saved
    """
    
    
    # Create coco categories list where id starts with 1
    coco_categories = []
    for i, cls in enumerate(annotation['categories']):
        category = {
            "id": i + 1, 
            "name": cls, 
            "supercategory": ""
        }
        coco_categories.append(category)
    
    # Create coco image list
    coco_images = []
    for i, image_name in enumerate(annotation['images']):
        
        # CHECK
        # if len(annotation['images'][image_name]['annotations']) == 0:
        #     continue
        
        img_id = i + 1
        width = annotation['images'][image_name]['width']
        height = annotation['images'][image_name]['height']
        
        image = {
            "id": img_id, 
            "width": width, 
            "height": height, 
            "file_name": image_name + image_ext,
            "license": 0, 
            "flickr_url": "", 
            "coco_url": "", 
            "date_captured": 0
        }
        coco_images.append(image)
    
    # Go through each bbox and save its data as coco annotations list
    coco_annotations = []
    bbox_id = 1
    
    for i, image_name in enumerate(annotation['images']):
        img_id = i + 1
        labeled_image = annotation['images'][image_name]
    
        for bbox in labeled_image['annotations']:
            x, y, w, h = bbox['bbox']
            cls_id = bbox['category_id']
            segmentation = bbox['segmentation']
            coco_annotation = {
                "id": bbox_id, 
                "image_id": img_id, 
                "category_id": cls_id + 1, 
                "segmentation": segmentation,
                "area": float(w * h), 
                "bbox": list(map(float, [x, y, w, h])), 
                "iscrowd": 0, 
                "attributes": {"occluded": False, "rotation": 0.0}
            }
            bbox_id += 1
            coco_annotations.append(coco_annotation)
    
    # Create default coco data
    licenses = [{"name": "", "id": 0, "url": ""}]
    info = {"contributor": "", 
            "date_created": "", 
            "description": "", 
            "url": "", 
            "version": "", 
            "year": ""}
    
    # Create coco dict and save
    coco = {
        'licenses': licenses,
        'info': info,
        'categories': coco_categories,
        'images': coco_images,
        'annotations': coco_annotations, 
    }

    with open(path, 'w') as f:
        json.dump(coco, f)



def write_yolo_det(annotation: dict, path: str):
    
    os.makedirs(path, exist_ok=True)
    
    for image_name in annotation['images']:
        bboxes = annotation['images'][image_name]['annotations']
        height = annotation['images'][image_name]['height']
        width = annotation['images'][image_name]['width']

        with open(os.path.join(path, image_name + '.txt'), 'w') as f:
            for bbox in bboxes:
                cls_id = bbox['category_id']
                if bbox['bbox_mode'] == 'xywhn':
                    x, y, w, h = bbox['bbox']
                elif bbox['bbox_mode'] == 'xywh':
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
    
