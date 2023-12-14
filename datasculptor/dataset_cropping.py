import cv2
import os
import numpy as np
from datasculptor import DetectionDataset, ISDataset, Annotation, AnnotatedImage, AnnotatedObject
from datasculptor import ImageSource, CropImageSource


def crop_dataset(dataset: DetectionDataset, size: tuple) -> DetectionDataset:
    image_sources = dataset.image_sources
    annotation = dataset.annotation
    categories = annotation.categories
    
    new_image_sources = []
    new_annotation = Annotation(categories=categories, images={})
    
    for img_src in image_sources:
        name = img_src.name
        if name not in annotation.images:
            continue
        
        lbl_image = annotation.images[name]
        cur_new_img_srcs, cur_new_lbl_images = crop_dataset_image(img_src, lbl_image, size)
        
        new_image_sources += cur_new_img_srcs
        new_annotation.images.update(cur_new_lbl_images)
    
    if type(dataset) == ISDataset:
        new_dataset = ISDataset(new_image_sources, new_annotation)    
    else:
        new_dataset = DetectionDataset(new_image_sources, new_annotation)
    return new_dataset
    
    
# TODO: subsets
def crop_dataset_image(image_source: ImageSource, labeled_image: AnnotatedImage, crop_size: tuple):
    new_img_srcs, new_lbl_images = [], {}
    width, height = labeled_image.width, labeled_image.height
    
    crop_w, crop_h = crop_size
    
    num_rows = -(-height // crop_h)
    num_cols = -(-width // crop_w)
    cnt = 0
    
    for row in range(num_rows):
        for col in range(num_cols):
            x2 = min(width, crop_w * (col + 1))
            y2 = min(height, crop_h * (row + 1))

            x1 = x2 - crop_w
            y1 = y2 - crop_h

            w, h = x2 - x1, y2 - y1
            
            new_img_src = CropImageSource(
                image_source,
                cnt,
                lambda x: crop_image(x, crop_size), 
                f"{image_source.name}_{cnt}",
            )
            new_img_srcs.append(new_img_src)
            
            new_bboxes = []
            for bbox in labeled_image.annotations:
                old_x, old_y, old_w, old_h = bbox.bbox
                if old_x >= x2 or old_y >= y2:
                    continue
                if old_x + old_w <= x1 or old_y + old_h <= y1:
                    continue
                
                new_x = max(0, old_x - x1)
                new_y = max(0, old_y - y1)
                new_x2 = min(w, new_x + old_w)
                new_y2 = min(h, new_y + old_h)
                new_w = new_x2 - new_x
                new_h = new_y2 - new_y
                
                # TODO: in another place 
                # if new_h + new_w < 10:
                #     continue
                
                segmentation = bbox.segmentation
                if len(segmentation) != 0:
                    new_segmentation = crop_segmentation(segmentation, (x1, y1, x2, y2))
                else:
                    new_segmentation = segmentation
                    
                new_bbox = {
                    'category_id': bbox.category_id,
                    'bbox': [new_x, new_y, new_w, new_h],
                    'bbox_mode': 'xywh',
                    'segmentation': new_segmentation,
                }
                new_bbox = AnnotatedObject(**new_bbox)
                new_bboxes.append(new_bbox)
            
            new_lbl_image = AnnotatedImage(width=w, height=h, annotations=new_bboxes)
            new_lbl_images[new_img_src.name] = new_lbl_image
            
            cnt += 1
    return new_img_srcs, new_lbl_images


def crop_image(img: np.ndarray, size: tuple):        
    width, height = img.shape[1], img.shape[0]
    
    crop_w, crop_h = size
    
    num_rows = -(-height // crop_h)
    num_cols = -(-width // crop_w)
    cnt = 0

    crops = []    
    for row in range(num_rows):
        for col in range(num_cols):
            
            x2 = min(width, crop_w * (col + 1))
            y2 = min(height, crop_h * (row + 1))

            x1 = x2 - crop_w
            y1 = y2 - crop_h

            crop = img[y1:y2, x1:x2]
            crops.append(crop)
    
    return crops


def crop_segmentation(segmentation, xyxy):
    x1, y1, x2, y2 = xyxy
    mask = np.zeros((y2 - y1, x2 - x1), dtype='uint8')
        
    for segment in segmentation:
        segment = np.array(segment)
        
        segment = segment.astype('int32')
        segment = segment.reshape(-1, 1, 2)
        
        segment[..., 0] -= x1
        segment[..., 1] -= y1
        
        cv2.fillPoly(mask, [segment], 255)
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_segmentation = []
    for cnt in contours:
        cnt = cnt.reshape(-1)
        new_segmentation.append(cnt.tolist())
    
    return new_segmentation
        
