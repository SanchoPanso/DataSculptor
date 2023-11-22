import os
import cv2
import numpy as np
import json
from typing import List, Dict, Union
from easydict import EasyDict


class AnnotatedObject(EasyDict):
    bbox: List[float]
    category_id: int
    segmentation: List[List[float]]
    
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {'bbox': [0] * 4, 'category_id': 0, 'segmentation': []}
        super().__init__(d, **kwargs)


class AnnotatedImage(EasyDict):
    height: int
    width: int
    annotations: List[AnnotatedObject]
    
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {'width': 0, 'height': 0, 'annotations': []}
        super().__init__(d, **kwargs)
    
def change_category_ids(image: AnnotatedImage, category_changes: Dict[int, int]) -> AnnotatedImage:
    """Update category indecies according to given map. 
    Indecies that is not mentioned in this map will be ignored

    :param category_changes: dict with key - old category id, value - new category i
    :return: 
    """
    
    new_image = AnnotatedImage(width=image.width, height=image.height)
    for bbox in image.annotations:
        category_id = bbox.category_id
        
        if category_id in category_changes:
            new_category_id = category_changes[category_id]
            bbox.category_id = new_category_id
        
        new_image.annotations.append(bbox)
    
    return new_image
    

class Annotation(EasyDict):
    """Class that contains annotations for computer vision tasks (detection, instance segmentation)
    
    :param categories: list of categories (or classes), defaults to None
    :param images: easydict of labeled images 
                   with image names as keys (without file extension), defaults to None
    """
    categories: List[str]
    images: EasyDict[str, AnnotatedImage]
    
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {'categories': [], 'images': {}}
        super().__init__(d, **kwargs)
    
    def __add__(self, other: 'Annotation') -> 'Annotation':
        additional_cats = [cat for cat in other.categories if cat not in self.categories]
        sum_cats = self.categories + additional_cats
        
        other_cat_changes = {}
        for i, other_cat in enumerate(other.categories):
            if other_cat not in additional_cats:
                continue
            other_cat_changes[i] = sum_cats.index(other_cat)
        
        sum_images = EasyDict(self.images.copy())
        unique_names = set(other.images.keys()) - set(self.images.keys())
        repeatable_names = set(other.images.keys()) - unique_names
        
        for name in unique_names:
            other_image = change_category_ids(other.images[name], other_cat_changes)
            sum_images[name] = other_image
        
        for name in repeatable_names:
            other_image = change_category_ids(other.images[name], other_cat_changes)
            sum_images[name] = AnnotatedImage(
                width=self.images[name].width, 
                height=self.images[name].height, 
                annotations=self.images[name].annotations + other_image.annotations)
        
        sum_annot = Annotation(categories=sum_cats, images=sum_images)        
        return sum_annot
    

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
        
        if type(coco_bbox['segmentation']) == list:
            segmentation = coco_bbox['segmentation']
        else:
            segmentation = rle2polygons(coco_bbox['segmentation'])
            
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
                
                if xc > 1 or yc > 1 or w > 1 or h > 1:
                    pass
                
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


def rle2polygons(rle: dict) -> list:
    size = rle['size']
    counts = rle['counts']
    
    mask = np.zeros((size[0] * size[1],), dtype='uint8')
    idx = 0
    color = 0
    
    for c in counts:
        mask[idx: idx + c] = color
        color = (color + 1) % 2
        idx += c
    
    mask = mask.reshape((size[0], size[1]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for cnt in contours:
        polygon = cnt.astype('int32').reshape(-1).tolist()
        polygons.append(polygon)
    
    return polygons


# annot = read_coco(r'C:\Users\HP\Downloads\360_17_0-6\annotations\instances_default.json')
# with open('annot.json', 'w') as f:
#     json.dump(annot, f)
    
