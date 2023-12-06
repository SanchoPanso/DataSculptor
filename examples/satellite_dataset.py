"""This script does the folowing
- take source aerial datasets, 
- crop them, 
- rename them, 
- change annotation classes
- remove empty images
- save as yolo dataset with only 'train' and 'valid' subsets

'data_path' has structure:

data_path/
- AAMDevelopedArea30cm_7_2/
    - annotations/
        - instances_default.json
    - images/
        - *.jpg
- Satellite30_1_3_98/
    ...
...

dataset_path - save path for result 

"""
import os
import cv2
import glob
import numpy as np
import cvml2
import json
import datetime
import rasterio
import zipfile
from pathlib import Path
from filesplit.split import Split


def main():
    data_path = r'D:\datasets\geo_ai\satellite\source'
    data_dirs = [
        'AAMDevelopedArea30cm_7_2',
        'Satellite30_1_3_98',
        'Satellite30_2_3_99',
        'Satellite30_6_3_51',
    ] 
    
    categories = ["palm_tree", "building", "farms", "trees"]

    datasets_dir = r'D:\datasets\geo_ai\sattelite\prepared'
    dataset_path = get_dataset_path(datasets_dir, 'geoai_satellite')
    print(dataset_path)
    dataset = cvml2.ISDataset()

    for cur_dir in data_dirs:
        src_images_dir = os.path.join(data_path, cur_dir, 'images')
        src_annot_path = os.path.join(data_path, cur_dir, 'annotations', 'instances_default.json')
        
        annot = cvml2.read_coco(src_annot_path)
        image_paths = glob.glob(os.path.join(src_images_dir, '*'))
        image_sources = cvml2.paths2image_sources(image_paths)
        
        cur_dataset = cvml2.ISDataset(image_sources, annot)
        cur_dataset.rename(lambda x: cur_dir + '_' + x)
        cur_dataset = cvml2.crop_dataset(cur_dataset, (1024, 1024))
        cur_dataset.annotation = change_annotation(cur_dataset.annotation, categories)
        cur_dataset.remove_empty_images()

        cur_dataset.split_by_proportions({'valid': 0.2, 'train': 0.8})
        
        dataset += cur_dataset
    
    dataset.install(
        dataset_path=dataset_path,
        dataset_name=os.path.basename(dataset_path),
        install_images=True,
        install_labels=True,
        install_description=True,
        image_ext='.png',
    )
    
    # Split the dataset into 999 MB blocks
    create_splitted_dataset(
        dataset_path,
        dataset_path,
        999*1024*1024,
    )


def get_dataset_path(datasets_dir: str, base_name: str):
    now = datetime.datetime.now()
    strf_date = now.strftime('%d%m%Y')
    name_with_date = f"{base_name}_{strf_date}"
    version_num = len(glob.glob(os.path.join(datasets_dir, name_with_date + '*')))
    name = f"{name_with_date}__v_{version_num}"
    
    return os.path.join(datasets_dir, name)


def change_annotation(annot: cvml2.Annotation, new_classes: list):
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
    

def delete_small_bboxes(annot: cvml2.Annotation):
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


def crop_images(src_dir, dst_dir, size):
    os.makedirs(dst_dir, exist_ok=True)
    crop_h, crop_w = size
    for img_fn in os.listdir(src_dir):
        print(img_fn)
        img = read_tiff(os.path.join(src_dir, img_fn))
        img_h, img_w = img.shape[:2]
        num_of_rows = img_h // crop_h if img_h % crop_h == 0 else img_h // crop_h + 1
        num_of_cols = img_w // crop_w if img_w % crop_w == 0 else img_w // crop_w + 1
        crop_cnt = 1
        
        for row in range(num_of_rows):
            for col in range(num_of_cols):
                
                crop_x1 = crop_w * col
                crop_y1 = crop_h * row

                crop_x2 = min(crop_x1 + crop_w, img_w)
                crop_y2 = min(crop_y1 + crop_h, img_h)
                
                crop = img[crop_y1: crop_y2, crop_x1: crop_x2]
                
                name, ext = os.path.splitext(img_fn)
                crop_fn = f'{name}_{crop_cnt}.jpg'
                cv2.imwrite(os.path.join(dst_dir, crop_fn), crop)
                crop_cnt += 1


def make_cropped_annotation(annotation: dict, size: tuple, isthing: list) -> dict:
    crop_h, crop_w = size
    classes = annotation['categories']

    new_annotation = {
        'categories': annotation['categories'],
        'images': {},
    }
    
    for img_name in annotation['images']:
        
        img_h = annotation['images'][img_name]['height']
        img_w = annotation['images'][img_name]['width']
        bboxes = annotation['images'][img_name]['annotations']
        
        num_of_rows = img_h // crop_h if img_h % crop_h == 0 else img_h // crop_h + 1
        num_of_cols = img_w // crop_w if img_w % crop_w == 0 else img_w // crop_w + 1
        crop_cnt = 1
        
        # Create semantic masks
        semantic_masks = {}
        for i in range(len(classes)):
            if not isthing[i]:
                mask = np.zeros((img_h, img_w), dtype='uint8')
                semantic_masks[i] = mask
        
        # Fill semantic masks with stuff objects
        for bbox in bboxes:            
            segmentation = bbox['segmentation']
            cls_id = bbox['category_id']
            
            if isthing[cls_id]:
                continue
            
            for s in segmentation:
                pts = np.array(s, dtype='int32').reshape((-1, 1, 2))
                cv2.fillPoly(semantic_masks[cls_id], [pts], 255)

        # Create crop annotation
        for row in range(num_of_rows):
            for col in range(num_of_cols):
                
                crop_name = f'{img_name}_{crop_cnt}'
                print(crop_name)
                new_annotation['images'][crop_name] = {
                    'width': crop_w,
                    'height': crop_h,
                    'annotations': [],
                }
                
                new_bboxes = []
                
                crop_x1 = crop_w * col
                crop_y1 = crop_h * row

                crop_x2 = min(crop_x1 + crop_w, img_w)
                crop_y2 = min(crop_y1 + crop_h, img_h)
                
                # Create bboxes for thing objects 
                for bbox in bboxes:
                    
                    segmentation = bbox['segmentation']
                    cls_id = bbox['category_id']
                    
                    if not isthing[cls_id]:
                        continue
                    
                    thing_mask = np.zeros((crop_y2 - crop_y1, crop_x2 - crop_x1), dtype='uint8')
                    for s in segmentation:
                        pts = np.array(s, dtype='int32').reshape((-1, 1, 2))
                        pts[:, 0, 0] -= crop_x1
                        pts[:, 0, 1] -= crop_y1
                        cv2.fillPoly(thing_mask, [pts], (255, 255, 255))
                    
                    contours, hierarchy = cv2.findContours(thing_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if len(contours) == 0:
                        continue
                    
                    c = contours[0]
                    
                    if cv2.contourArea(c) < 10:
                        continue
                    
                    x1 = int(c[:, 0, 0].min())
                    y1 = int(c[:, 0, 1].min())
                    x2 = int(c[:, 0, 0].max())
                    y2 = int(c[:, 0, 1].max())
                    w = x2 - x1
                    h = y2 - y1
                    
                    new_segmentation = c.reshape((1, -1)).tolist()
                    
                    new_bbox = {
                        'category_id': cls_id,
                        'bbox': [x1, y1, w, h],
                        'bbox_mode': 'xywh',
                        'segmentation': new_segmentation,
                    }
                    new_bboxes.append(new_bbox)
                
                # Create bboxes for stuff objects
                for cls_id in range(len(classes)):
                    
                    if isthing[cls_id]:
                        continue
                    
                    semantic_crop = semantic_masks[cls_id][crop_y1: crop_y2, crop_x1: crop_x2]         
                    contours, hierarchy = cv2.findContours(semantic_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                    for c in contours:
                        
                        if cv2.contourArea(c) < 10:
                            continue
                        
                        x1 = int(c[:, 0, 0].min())
                        y1 = int(c[:, 0, 1].min())
                        x2 = int(c[:, 0, 0].max())
                        y2 = int(c[:, 0, 1].max())
                        w = x2 - x1
                        h = y2 - y1
                        
                        new_segmentation = c.reshape((1, -1)).tolist()
                        
                        new_bbox = {
                            'category_id': cls_id,
                            'bbox': [x1, y1, w, h],
                            'bbox_mode': 'xywh',
                            'segmentation': new_segmentation,
                        }
                        new_bboxes.append(new_bbox)
                
                
                new_annotation['images'][crop_name]['annotations'] = new_bboxes
                crop_cnt += 1
    
    return new_annotation


def filter_mask(annotation: dict, image_dir: str, filtering_cls_id: int):
    
    for img_name in annotation['images']:
        
        image_paths = glob.glob(os.path.join(image_dir, img_name + '*'))
        if len(image_paths) == 0:
            continue
        img = cv2.imread(image_paths[0])[:, :, 1]#, cv2.IMREAD_GRAYSCALE)
        
        img_h = annotation['images'][img_name]['height']
        img_w = annotation['images'][img_name]['width']
        bboxes = annotation['images'][img_name]['annotations']
        
        mask = np.zeros((img_h, img_w), dtype='uint8')
        new_bboxes = []
        
        for bbox in bboxes:
            segmentation = bbox['segmentation']
            cls_id = bbox['category_id']
            
            if cls_id != filtering_cls_id:
                new_bboxes.append(bbox)
                continue
            
            for s in segmentation:
                pts = np.array(s, dtype='int32').reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 1)
        
        if mask.sum() == 0:
            annotation['images'][img_name]['annotations'] = new_bboxes
            continue
        
        thresh = int((img * mask).sum() / mask.sum() * 0.8)
        ret, f_mask = cv2.threshold(img, thresh, 1, cv2.THRESH_BINARY_INV)
        
        res_mask = mask * f_mask
        contours, hierarchy = cv2.findContours(res_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            
            if cv2.contourArea(c) < 50:
                continue
            
            x1 = int(c[:, 0, 0].min())
            y1 = int(c[:, 0, 1].min())
            x2 = int(c[:, 0, 0].max())
            y2 = int(c[:, 0, 1].max())
            w = x2 - x1
            h = y2 - y1
            
            new_segmentation = c.reshape((1, -1)).tolist()

            new_bbox = {
                'category_id': filtering_cls_id,
                'bbox': [x1, y1, w, h],
                'bbox_mode': 'xywh',
                'segmentation': new_segmentation,
            }
            new_bboxes.append(new_bbox)
        annotation['images'][img_name]['annotations'] = new_bboxes
    
    return annotation

def read_tiff(path):
    dataset = rasterio.open(path)

    bands = []
    for i in [3, 2, 1]:
        band = dataset.read(i)
        bands.append(band)
        
    img = cv2.merge(bands)
    return img
        

if __name__ == '__main__':
    main()
