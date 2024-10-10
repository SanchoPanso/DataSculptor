import os
import cv2
import glob
import numpy as np
import json
import datetime
import rasterio
import zipfile
from pathlib import Path
from filesplit.split import Split
from shapely import Polygon, MultiPolygon, union_all
from typing import List, Dict

from datasculptor.annotation import Annotation, AnnotatedImage, AnnotatedObject


def merge_and_split(annot: Annotation, category_name: str):
    category_id = annot.categories.index(category_name)

    for img_name in annot.images:
        img = annot.images[img_name]

        mask = np.zeros((img.height, img.width), dtype=np.uint8)
        new_objs = []
        for obj in img.annotations:
            if obj.category_id != category_id:
                new_objs.append(obj)
                continue

            for segment in obj.segmentation:
                segment = np.array(segment)
                segment = segment.astype('int32')
                segment = segment.reshape(-1, 1, 2)
                cv2.fillPoly(mask, [segment], 255)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            segment = cnt.reshape(-1).tolist()
            obj = AnnotatedObject([x, y, w, h], category_id, [segment])
            new_objs.append(obj)
        
        img.annotations = new_objs
    return annot



def get_dataset_path(datasets_dir: str, base_name: str):
    now = datetime.datetime.now()
    strf_date = now.strftime('%d%m%Y')
    name_with_date = f"{base_name}_{strf_date}"
    version_num = len(glob.glob(os.path.join(datasets_dir, name_with_date + '*')))
    name = f"{name_with_date}__v_{version_num}"
    
    return os.path.join(datasets_dir, name)


def change_annotation(annot: Annotation, new_classes: list):
    classes = annot.categories
    
    conformity = {}
    for i in range(len(classes)):
        if classes[i] in new_classes:
            conformity[i] = new_classes.index(classes[i])
    
    images = annot.images
    for name in images:
        new_bboxes = []
        
        for bbox in images[name].annotations:
            if bbox.category_id not in conformity:
                continue
            
            bbox.category_id = conformity[bbox.category_id]
            new_bboxes.append(bbox)
        images[name].annotations = new_bboxes
        
    annot.categories = new_classes
    return annot
    

def delete_small_bboxes(annot: Annotation):
    images = annot.images
    for name in images:
        new_bboxes = []
        
        for bbox in images[name].annotations:
            if annot.categories[bbox.category_id] != 'household_garbage':
                continue
        
            x, y, w, h = bbox['bbox']
            if w > 20 and h > 20:
                bbox.category_id = 0
                new_bboxes.append(bbox)
        images[name].annotations = new_bboxes
        
    annot.categories = ['household_garbage']
    return annot
    

def create_splitted_dataset(src_dir: str, dst_dir: str, block_volume: int):
    
    os.makedirs(dst_dir, exist_ok=True)
    
    for sample_name in ['train', 'valid']:
        sample_path = os.path.join(src_dir, sample_name)
        
        with zipfile.ZipFile(f"{sample_path}.zip", mode="w") as archive:
            directory = Path(sample_path)
            for file_path in directory.rglob("*"):
                archive.write(file_path, arcname=file_path.relative_to(directory))

        split = Split(f"{sample_path}.zip", dst_dir)
        split.bysize(block_volume)


def read_tiff(path):
    dataset = rasterio.open(path)

    bands = []
    for i in [3, 2, 1]:
        band = dataset.read(i)
        bands.append(band)
        
    img = cv2.merge(bands)
    return img


def join_splitted_annotations(
    annotated_images: List[List[AnnotatedImage]],
    orig_img_size: tuple,
    overlap: int = 0,
) -> AnnotatedImage:
    
    def find_coord(row, col, crop_w, crop_h, left_pad, upper_pad, overlap):
        x1 = col * (crop_w - overlap) - left_pad
        y1 = row * (crop_h - overlap) - upper_pad
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        return x1, y1, x2, y2
    
    num_rows, num_cols = len(annotated_images), len(annotated_images[0])
    crop_h, crop_w = annotated_images[0][0].height, annotated_images[0][0].width
    
    width, height = orig_img_size
    
    padded_height = num_rows * (crop_h - overlap) + overlap
    padded_width = num_cols * (crop_w - overlap) + overlap
    
    left_pad = (padded_width - width) // 2
    upper_pad = (padded_height - height) // 2
    
    common_obj = []
    overlapped_obj_ids = {}

    for row in range(num_rows):
        for col in range(num_cols):
            
            x1, y1, x2, y2 = find_coord(row, col, crop_w, crop_h, left_pad, upper_pad, overlap)
            objs = annotated_images[row][col].annotations
            objs = add_offset_to_objs(objs, x1, y1)
            
            # Right neighbour
            if col + 1 != num_cols:
                right_objs = annotated_images[row][col + 1].annotations
                xr1, yr1, _, _ = find_coord(row, col + 1, crop_w, crop_h, left_pad, upper_pad, overlap)
                right_objs = add_offset_to_objs(right_objs, xr1, yr1)
                right_overlapped_objs, cur_ol_ids, right_ol_ids = join_overlapped_obj(objs, right_objs)
                common_obj += right_overlapped_objs
                
                if f"{row}_{col}" not in overlapped_obj_ids:
                    overlapped_obj_ids[f"{row}_{col}"] = []
                overlapped_obj_ids[f"{row}_{col}"] += cur_ol_ids

                if f"{row}_{col + 1}" not in overlapped_obj_ids:
                    overlapped_obj_ids[f"{row}_{col + 1}"] = []
                overlapped_obj_ids[f"{row}_{col + 1}"] += right_ol_ids

            # Bottom neighbour
            if row + 1 != num_rows:
                bottom_objs = annotated_images[row + 1][col].annotations
                xb1, yb1, _, _ = find_coord(row + 1, col, crop_w, crop_h, left_pad, upper_pad, overlap)
                bottom_objs = add_offset_to_objs(bottom_objs, xb1, yb1)
                bottom_overlapped_objs, cur_ol_ids, bottom_ol_ids = join_overlapped_obj(objs, bottom_objs)
                common_obj += bottom_overlapped_objs

                if f"{row}_{col}" not in overlapped_obj_ids:
                    overlapped_obj_ids[f"{row}_{col}"] = []
                overlapped_obj_ids[f"{row}_{col}"] += cur_ol_ids

                if f"{row + 1}_{col}" not in overlapped_obj_ids:
                    overlapped_obj_ids[f"{row + 1}_{col}"] = []
                overlapped_obj_ids[f"{row + 1}_{col}"] += bottom_ol_ids
            
            nonoverlapped_obj = [obj for i, obj in enumerate(annotated_images[row][col].annotations)
                                 if i not in overlapped_obj_ids[f"{row}_{col}"]]

            common_obj += nonoverlapped_obj
            
    common_image = AnnotatedImage(width, height, common_obj)
    return common_image


def add_offset_to_objs(objs: List[AnnotatedObject], x1: float, y1: float):
    new_objs = []
    for obj in objs:
        x, y, w, h = obj.bbox
        new_bbox = [x + x1, y + y1, w, h]

        new_segmentation = []
        if len(obj.segmentation) != 0:
            for segment in obj.segmentation:
                segment = np.array(segment).reshape(-1, 1, 2)
                segment[..., 0] += x1
                segment[..., 1] += y1
                segment = segment.reshape(-1).tolist()
                new_segmentation.append(segment)

        new_obj = AnnotatedObject(new_bbox, obj.category_id, new_segmentation)
        new_objs.append(new_obj)
    
    return new_objs


def join_overlapped_obj(objs1: List[AnnotatedObject], objs2: List[AnnotatedObject]):
    overlapped_objs = []
    overlapped_ids_1 = []
    overlapped_ids_2 = []

    for i, obj1 in enumerate(objs1):
        for j, obj2 in enumerate(objs2):
            overlapped_obj = get_obj_union(obj1, obj2)
            if overlapped_obj is None:
                continue
            
            overlapped_objs.append(overlapped_obj)
            overlapped_ids_1.append(i)
            overlapped_ids_2.append(j)
    
    return overlapped_objs, overlapped_ids_1, overlapped_ids_2


def get_obj_union(obj1: AnnotatedObject, obj2: AnnotatedObject) -> AnnotatedObject | None:
    if len(obj1.segmentation) == 0 or len(obj2.segmentation) == 0:
        return None
    
    if obj1.category_id != obj2.category_id:
        return None
    
    obj1_polys = [Polygon(np.array(s).reshape(-1, 2)).buffer(0) for s in obj1.segmentation]
    obj2_polys = [Polygon(np.array(s).reshape(-1, 2)).buffer(0) for s in obj2.segmentation]

    obj1_multipoly = MultiPolygon(obj1_polys)
    obj2_multipoly = MultiPolygon(obj2_polys)

    if not obj1_multipoly.intersects(obj2_multipoly):
        return None
    
    union_geometry = obj1_multipoly.union(obj2_multipoly)

    if type(union_geometry) == Polygon:
        union_multipoly = MultiPolygon([union_geometry])
    elif type(union_geometry) == MultiPolygon:
        union_multipoly = union_geometry
    else:
        return None
    
    new_segmentation = []

    x1, y1, x2, y2 = -1, -1, -1, -1

    for i, poly in enumerate(union_multipoly.geoms):
        xs, ys = poly.exterior.coords.xy
        xs, ys = xs.tolist(), ys.tolist()

        new_segment = np.array([xs, ys]).T.reshape(-1).tolist()[:-2]
        new_segmentation.append(new_segment)

        if i == 0:
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
        else:
            x1, x2 = min([x1] + xs), max([x2] + xs)
            y1, y2 = min([y1] + ys), max([y2] + ys)
    
    new_bbox = [x1, y1, x2 - x1, y2 - y1]
    union_obj = AnnotatedObject(new_bbox, obj1.category_id, new_segmentation)
    return union_obj

