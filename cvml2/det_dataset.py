import os
import sys
import cv2
import random
import math
import numpy as np
from abc import ABC
from typing import List, Set, Dict, Callable
from enum import Enum
import logging
import time

from cvml2.annotation import write_yolo_det
from cvml2.image_source import ImageSource


class DetectionDataset:
    """
    Class of dataset, representing sets of labeled images with bounding boxes, 
    which are used in detection tasks.
    """

    def __init__(self, 
                 image_sources: List[ImageSource] = None,
                 annotation: dict = None, 
                 samples: Dict[str, List[int]] = None):
        """Constructor

        :param image_sources: list image sources, representing images, that will be placed in dataset, defaults to None
        :param annotation: annotation of images, represented by image sources, defaults to None
        :param samples: dict of lists of indexes of images, that corresponds to a specific set, defaults to None
        """
        
        self.image_sources = image_sources or []
        self.annotation = annotation or {}
        self.samples = samples or {}
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
    

    def __len__(self):
        return len(self.image_sources)

    def __getitem__(self, item):
        return self.image_sources[item]

    def __add__(self, other):
        
        # Addition of image sources
        sum_image_sources = self.image_sources + other.image_sources
        
        # Addition of annotation
        sum_annotation = self.annotation + other.annotation
        
        # Addition of samples
        self_sample_names = set(self.samples.keys())
        other_sample_names = set(other.samples.keys())
        
        # sum_sample_names - union of two sample names 
        sum_sample_names = self_sample_names or other_sample_names
        sum_samples = {}
        
        # In new samples self indexes remain their values, others - are addicted with number of images in self
        # (other images addict to the end of common list) 
        for name in sum_sample_names:
            sum_samples[name] = []
            if name in self_sample_names:
                sum_samples[name] += self.samples[name]
            if name in other_sample_names:
                sum_samples[name] += list(map(lambda x: x + len(self), other.samples[name]))
        
        return DetectionDataset(sum_image_sources, sum_annotation, sum_samples)
    
    def rename(self, rename_callback: Callable):
        
        for i in range(len(self.image_sources)):
            
            # Rename image source
            old_name = self.image_sources[i].name
            new_name = rename_callback(self.image_sources[i].name)
            self.image_sources[i].name = new_name
            
            # Rename in bbox_map
            if old_name in self.annotation['images']:
                image_info = self.annotation.bbox_map[old_name]
                self.annotation.bbox_map.pop(old_name)
                self.annotation.bbox_map[new_name] = image_info
            # else:
            #     self.annotation.bbox_map[new_name] = []
            

    def split_by_proportions(self, proportions: dict):
        all_idx = [i for i in range(len(self.image_sources))]
        random.shuffle(all_idx)

        length = len(self.image_sources)
        split_start_idx = 0
        split_end_idx = 0

        # Reset current split indexes
        self.samples = {}

        num_of_names = len(proportions.keys())

        for i, split_name in enumerate(proportions.keys()):
            split_end_idx += math.ceil(proportions[split_name] * length)
            self.samples[split_name] = all_idx[split_start_idx: split_end_idx]
            split_start_idx = split_end_idx

            if i + 1 == num_of_names and split_end_idx < len(all_idx):
                self.samples[split_name] += all_idx[split_end_idx: len(all_idx)]
        
        # logging
        message = "In dataset the following splits was created: "
        for i, split_name in enumerate(self.samples.keys()):
            message += f"{split_name}({len(self.samples[split_name])})"
            if i != len(self.samples.keys()) - 1:
                message += ", "
        self.logger.info(message)

    def split_by_dataset(self, yolo_dataset_path: str):

        # Define names of splits as dirnames in dataset directory
        split_names = [name for name in os.listdir(yolo_dataset_path)
                       if os.path.isdir(os.path.join(yolo_dataset_path, name))]

        # Reset current split indexes
        self.samples = {}

        for split_name in split_names:

            # Place names of orig dataset split in set structure
            orig_dataset_files = os.listdir(os.path.join(yolo_dataset_path, split_name, 'labels'))
            orig_names_set = set()

            for file in orig_dataset_files:
                name, ext = os.path.splitext(file)
                orig_names_set.add(name)

            # If new_name in orig dataset split then update split indexes of current dataset
            self.samples[split_name] = []
            for i, image_source in enumerate(self.image_sources):
                new_name = image_source.name
                if new_name in orig_names_set:
                    self.samples[split_name].append(i)
    
    def add_with_proportion(self, dataset, proportions: dict):
        
        assert proportions.keys() == self.samples.keys()
        
        orig_length = len(self)
        dataset_length = len(dataset)
        result_length = orig_length + dataset_length
        
        dataset_proportions = {}
        for name in self.samples:
            orig_sample_length = len(self.samples[name])
            result_sample_length = proportions[name] * result_length
            dataset_proportions[name] = (result_sample_length - orig_sample_length) / dataset_length
        
        dataset.split_by_proportions(dataset_proportions)
        new_dataset = self + dataset
        
        # logging
        message = "Create summary dataset with samples: "
        for i, split_name in enumerate(self.samples.keys()):
            message += f"{split_name}({len(self.samples[split_name])})"
            if i != len(self.samples.keys()) - 1:
                message += ", "
        self.logger.info(message)
        
        return new_dataset

    def install(self, 
                dataset_path: str,
                dataset_name: str = 'dataset',
                image_ext: str = 'jpg', 
                install_images: bool = True, 
                install_labels: bool = True, 
                # install_annotations: bool = True, 
                install_description: bool = True):
        
        for split_name in self.samples.keys():
            split_ids = self.samples[split_name]    
            
            if install_images:
                images_dir = os.path.join(dataset_path, split_name, 'images')
                os.makedirs(images_dir, exist_ok=True)
                
                for i, split_idx in enumerate(split_ids):
                    image_source = self.image_sources[split_idx] 
                    image_source.save(os.path.join(images_dir, image_source.name + image_ext))                
                    self.logger.info(f"[{i + 1}/{len(split_ids)}] " + 
                                     f"{split_name}:{self.image_sources[i].name}{image_ext} is done")
                self.logger.info(f"{split_name} is done")

            if install_labels:
                labels_dir = os.path.join(dataset_path, split_name, 'labels')
                os.makedirs(labels_dir, exist_ok=True)
                sample_annotation = self._get_sample_annotation(split_name)
                write_yolo_det(sample_annotation, labels_dir)
                self.logger.info(f"{split_name}:yolo_labels is done")
            
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
    
    
    def _get_sample_annotation(self, sample_name: str) -> dict:
        sample_classes = self.annotation['categories']
        sample_images = {}
        
        for i in self.samples[sample_name]:
            image_source = self.image_sources[i]
            name = image_source.name
            sample_images[name] = self.annotation['images'][name]
        
        sample_annotation = {'categories': sample_classes, 'images': sample_images}
        return sample_annotation
    
    def _write_description(self, path: str, dataset_name: str):
        text = f"train: {dataset_name}/train/images\n" \
               f"val: {dataset_name}/valid/images\n\n" \
               f"nc: {len(self.annotation['categories'])}\n" \
               f"names: {self.annotation['categories']}"
        with open(path, 'w') as f:
            f.write(text)
        






