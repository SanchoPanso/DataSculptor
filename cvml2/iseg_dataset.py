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

from cvml2.annotation import write_yolo_iseg
from cvml2.det_dataset import DetectionDataset
from cvml2.image_source import ImageSource


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
        
        # Addition of image sources
        sum_image_sources = self.image_sources + other.image_sources
        
        # Addition of annotation
        # TODO: repeatable image processing
        sum_annotation = self.annotation
        self.annotation['images'].update(other.annotation['images'])
        
        # Addition of susets
        self_sample_names = set(self.subsets.keys())
        other_sample_names = set(other.subsets.keys())
        
        # sum_sample_names - union of two sample names 
        sum_sample_names = self_sample_names or other_sample_names
        sum_samples = {}
        
        # In new samples self indexes remain their values, others - are addicted with number of images in self
        # (other images addict to the end of common list) 
        for name in sum_sample_names:
            sum_samples[name] = []
            if name in self_sample_names:
                sum_samples[name] += self.subsets[name]
            if name in other_sample_names:
                sum_samples[name] += list(map(lambda x: x + len(self), other.subsets[name]))
        
        return ISDataset(sum_image_sources, sum_annotation, sum_samples)
    

    def install(self, 
                dataset_path: str,
                dataset_name: str = 'dataset',
                image_ext: str = '.jpg', 
                install_images: bool = True, 
                install_labels: bool = True, 
                # install_annotations: bool = True, 
                install_description: bool = True):
        
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
            
            # if install_annotations:
            #     annotation_dir = os.path.join(dataset_path, split_name, 'annotations')
            #     os.makedirs(annotation_dir, exist_ok=True)
            #     coco_path = os.path.join(annotation_dir, 'data.json')
            #     sample_annotation = self._get_sample_annotation(split_name)
            #     write_coco(sample_annotation, coco_path)
            #     self.logger.info(f"{split_name}:coco_annotation is done")
            
        if install_description:
            self._write_description(os.path.join(dataset_path, 'data.yaml'), dataset_name)
            self.logger.info(f"Description is done")


