from __future__ import annotations
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
import shutil

from datasculptor import Annotation, write_yolo_det
from datasculptor import ImageSource, paths2image_sources


class DetectionDataset:
    """
    Class of dataset, representing sets of labeled images with bounding boxes, 
    which are used in detection tasks.
    """

    image_sources: List[ImageSource]
    annotation: Annotation
    subsets: Dict[str, int]

    def __init__(self,
                 image_sources: List[ImageSource] = None,
                 annotation: Annotation = None,
                 subsets: Dict[str, List[int]] = None):
        """Constructor

        :param image_sources: list image sources, representing images, that will be placed in dataset, defaults to None
        :param annotation: annotation of images, represented by image sources, defaults to None
        :param samples: dict of lists of indexes of images, that corresponds to a specific set, defaults to None
        """

        self.image_sources = image_sources or []
        self.annotation = annotation or Annotation()
        self.subsets = subsets or {}

        self.resizer = None

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)


    def __len__(self):
        return len(self.image_sources)

    def __getitem__(self, item):
        return self.image_sources[item]

    def __add__(self, other: DetectionDataset):

        # Addition of image sources
        sum_image_sources = self.image_sources + other.image_sources

        # Addition of annotation
        # sum_annotation = self.annotation
        # self.annotation['images'].update(other.annotation['images'])
        sum_annotation = self.annotation + other.annotation

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

        return DetectionDataset(sum_image_sources, sum_annotation, sum_samples)

    def resize(self, size: tuple):
        assert len(size) == 2

        # Add resize fn to image sources
        for img_src in self.image_sources:
            img_src.preprocessing_fns.append(lambda x: cv2.resize(x, size))

        # Go through annotation and correct coordinates
        new_width, new_height = size
        for image_name in self.annotation.images:
            labeled_image = self.annotation.images[image_name]

            old_width = labeled_image.width
            old_height = labeled_image.height

            # Correct image size
            labeled_image.width = new_width
            labeled_image.height = new_height

            # Correct bbox coordinates of cur image
            for bbox in labeled_image.annotations:
                x, y, w, h = bbox.bbox

                x *= new_width / old_width
                w *= new_width / old_width
                y *= new_height / old_height
                h *= new_height / old_height

                bbox.bbox = [x, y, w, h]


    def rename(self, rename_callback: Callable):

        for i in range(len(self.image_sources)):

            # Rename image sources
            old_name = self.image_sources[i].name
            new_name = rename_callback(old_name)
            self.image_sources[i].name = new_name

            # Rename annotations
            if old_name in self.annotation.images:
                image_info = self.annotation.images[old_name]
                self.annotation.images.pop(old_name)
                self.annotation.images[new_name] = image_info

    def split_by_proportions(self, proportions: dict):
        all_idx = [i for i in range(len(self.image_sources))]
        random.shuffle(all_idx)

        length = len(self.image_sources)
        split_start_idx = 0
        split_end_idx = 0

        # Reset current split indexes
        self.subsets = {}

        num_of_names = len(proportions.keys())

        for i, split_name in enumerate(proportions.keys()):
            split_end_idx += math.ceil(proportions[split_name] * length)
            self.subsets[split_name] = all_idx[split_start_idx: split_end_idx]
            split_start_idx = split_end_idx

            if i + 1 == num_of_names and split_end_idx < len(all_idx):
                self.subsets[split_name] += all_idx[split_end_idx: len(all_idx)]

        # logging
        message = "In dataset the following splits was created: "
        for i, split_name in enumerate(self.subsets.keys()):
            message += f"{split_name}({len(self.subsets[split_name])})"
            if i != len(self.subsets.keys()) - 1:
                message += ", "
        self.logger.info(message)

    def split_by_dataset(self, yolo_dataset_path: str):

        # Define names of splits as dirnames in dataset directory
        split_names = [name for name in os.listdir(yolo_dataset_path)
                       if os.path.isdir(os.path.join(yolo_dataset_path, name))]

        # Reset current split indexes
        self.subsets = {}

        for split_name in split_names:

            # Place names of orig dataset split in set structure
            orig_dataset_files = os.listdir(os.path.join(yolo_dataset_path, split_name, 'labels'))
            orig_names_set = set()

            for file in orig_dataset_files:
                name, ext = os.path.splitext(file)
                orig_names_set.add(name)

            # If new_name in orig dataset split then update split indexes of current dataset
            self.subsets[split_name] = []
            for i, image_source in enumerate(self.image_sources):
                new_name = image_source.name
                if new_name in orig_names_set:
                    self.subsets[split_name].append(i)

    def add_with_proportion(self, dataset, proportions: dict):

        assert proportions.keys() == self.subsets.keys()

        orig_length = len(self)
        dataset_length = len(dataset)
        result_length = orig_length + dataset_length

        dataset_proportions = {}
        for name in self.subsets:
            orig_sample_length = len(self.subsets[name])
            result_sample_length = proportions[name] * result_length
            dataset_proportions[name] = (result_sample_length - orig_sample_length) / dataset_length

        dataset.split_by_proportions(dataset_proportions)
        new_dataset = self + dataset

        # logging
        message = "Create summary dataset with samples: "
        for i, split_name in enumerate(self.subsets.keys()):
            message += f"{split_name}({len(self.subsets[split_name])})"
            if i != len(self.subsets.keys()) - 1:
                message += ", "
        self.logger.info(message)

        return new_dataset

    def remove_empty_images(self):
        new_img_srcs = []
        for img_src in self.image_sources:
            name = img_src.name
            if name not in self.annotation.images:
                continue

            bboxes = self.annotation.images[name].annotations
            if len(bboxes) == 0:
                continue

            new_img_srcs.append(img_src)
        self.image_sources = new_img_srcs

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
                write_yolo_det(sample_annotation, labels_dir)
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

        self._clear_cache(dataset_path)


    def _get_sample_annotation(self, sample_name: str) -> dict:
        sample_classes = self.annotation.categories
        sample_images = {}

        for i in self.subsets[sample_name]:
            name = self.image_sources[i].name
            if name not in self.annotation.images:
                continue
            sample_images[name] = self.annotation.images[name]

        sample_annotation = {'categories': sample_classes, 'images': sample_images}
        return sample_annotation

    def _write_description(self, path: str, dataset_name: str):
        text = f"train: {dataset_name}/train/images\n" \
               f"val: {dataset_name}/valid/images\n\n" \
               f"nc: {len(self.annotation.categories)}\n" \
               f"names: {self.annotation.categories}"
        with open(path, 'w') as f:
            f.write(text)

    def _clear_cache(self, dataset_path):
        shutil.rmtree(os.path.join(dataset_path, '.cvml2_cache'))







