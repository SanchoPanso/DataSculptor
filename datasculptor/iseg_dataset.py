import os
import cv2
import random
import logging
import math
import numpy as np
from itertools import groupby
from abc import ABC
from typing import List, Set, Dict, Callable
from enum import Enum

from datasculptor.annotation import write_yolo_iseg, write_coco
from datasculptor.det_dataset import DetectionDataset
from datasculptor.image_source import ImageSource


# def convert_mask_to_coco_rle(color_mask: np.ndarray, bbox: BoundingBox) -> dict:
#     x, y, w, h = map(int, bbox.get_absolute_bounding_box())
#     width, height = color_mask.shape[:2]

#     rle = {
#         'size': [width, height],
#         'counts': [],
#     }

#     x = min(x, width)
#     y = min(y, height)
#     w = min(w, width - x)
#     h = min(h, height - y)

#     if w == 0 or h == 0:
#         rle['counts'] = [0]
#         return rle

#     obj_crop = color_mask[y: y + h, x: x + w]
#     obj_crop = cv2.cvtColor(obj_crop, cv2.COLOR_BGR2GRAY)
#     ret, binary_obj_crop = cv2.threshold(obj_crop, 1, 1, cv2.THRESH_BINARY)

#     binary_mask = np.zeros((height, width), dtype='uint8')
#     binary_mask[y: y + h, x: x + w] = binary_obj_crop

#     counts = []

#     for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
#         if i == 0 and value == 1:
#             counts.append(0)
#         counts.append(len(list(elements)))
    
#     rle['counts'] = counts 

#     return rle



class ISDataset(DetectionDataset):
    """
    Class of dataset, representing sets of labeled images with bounding boxes, 
    which are used in instance segmentation tasks.
    """

    def __init__(self, 
                 image_sources: List[ImageSource] = None,
                 annotation: dict = None, 
                 samples: Dict[str, List[int]] = None):
    
        super(ISDataset, self).__init__(image_sources, annotation, samples)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    def __add__(self, other):
        
        sum_dataset = super().__add__(other)
        
        return ISDataset(sum_dataset.image_sources, sum_dataset.annotation, sum_dataset.subsets)
    
    
    def resize(self, size: tuple):
        assert len(size) == 2
        
        # Add resize fn to image sources
        for img_src in self.image_sources:
            img_src.preprocessing_fns.append(lambda x: cv2.resize(x, size))
        
        # Go through annotation and correct coordinates
        new_width, new_height = size
        for image_name in self.annotation.images:
            labeled_image = self.annotation.images[image_name]
            
            old_width = labeled_image['width']
            old_height = labeled_image['height']
            
            # Correct image size
            labeled_image['width'] = new_width
            labeled_image['height'] = new_height
            
            # Correct bbox coordinates of cur image
            for bbox in labeled_image['annotations']:
                x, y, w, h = bbox['bbox']
                
                x *= new_width / old_width
                w *= new_width / old_width
                y *= new_height / old_height
                h *= new_height / old_height
                
                bbox['bbox'] = [x, y, w, h]
                
                segmentation = bbox['segmentation']
                for segment in segmentation:
                    segment = np.array(segment).astype('float64').reshape(-1, 1, 2)
                    segment[..., 0] *= new_width / old_width
                    segment[..., 1] *= new_height / old_height
                    segment = segment.reshape(-1).astype('int32').tolist()
    
    
    def remove_empty_images(self):
        new_img_srcs = []
        for img_src in self.image_sources:
            name = img_src.name
            if name not in self.annotation.images:
                continue
            
            bboxes = self.annotation.images[name]['annotations']
            bboxes_is_empty = False
            for bbox in bboxes:
                if len(bbox.segmentation) == 0:
                    bboxes_is_empty = True
                    break
            if bboxes_is_empty:
                continue
            
            new_img_srcs.append(img_src)
        self.image_sources = new_img_srcs

    def install(self, 
                dataset_path: str,
                dataset_name: str = 'dataset',
                image_ext: str = '.jpg', 
                install_images: bool = True, 
                install_labels: bool = True, 
                install_annotations: bool = False, 
                install_description: bool = True):
        
        os.makedirs(dataset_path, exist_ok=True)
        
        for subset_name in self.subsets.keys():
            subset_ids = self.subsets[subset_name]    
            
            if install_images:
                images_dir = os.path.join(dataset_path, subset_name, 'images')
                os.makedirs(images_dir, exist_ok=True)
                
                for i, split_idx in enumerate(subset_ids):
                    # image_source = self.image_sources[split_idx] 
                    # img = cv2.imread(image_source)
                    
                    if self.resizer is not None:
                        img = self.resizer(img)
                    
                    image_source = self.image_sources[split_idx] 
                    # save_img_path = os.path.join(images_dir, image_source.name + image_ext)
                    image_source.save(images_dir, image_ext, cache_dir=os.path.join(dataset_path, '.cvml2_cache'))                
                    
                    self.logger.info(f"[{i + 1}/{len(subset_ids)}] " + 
                                     f"{subset_name}:{self.image_sources[split_idx].name}{image_ext} is done")
                self.logger.info(f"{subset_name} is done")

            if install_labels:
                labels_dir = os.path.join(dataset_path, subset_name, 'labels')
                os.makedirs(labels_dir, exist_ok=True)
                sample_annotation = self._get_sample_annotation(subset_name)
                write_yolo_iseg(sample_annotation, labels_dir)
                self.logger.info(f"{subset_name}:yolo_labels is done")
            
            if install_annotations:
                annotation_dir = os.path.join(dataset_path, subset_name, 'annotations')
                os.makedirs(annotation_dir, exist_ok=True)
                coco_path = os.path.join(annotation_dir, 'data.json')
                sample_annotation = self._get_sample_annotation(subset_name)
                write_coco(sample_annotation, coco_path, image_ext)
                self.logger.info(f"{subset_name}:coco_annotation is done")
            
        if install_description:
            self._write_description(os.path.join(dataset_path, 'data.yaml'), dataset_name)
            self.logger.info(f"Description is done")


