import os
import sys
import cv2
import pytest
import shutil
import numpy as np
from pathlib import Path

from datasculptor import Annotation, AnnotatedImage, AnnotatedObject
from datasculptor import Dataset
from datasculptor import read_yolo
from datasculptor import crop_dataset
from datasculptor import paths2image_sources


def test_crop_2x2_default():
    """_summary_
    
    Polygons schema:
    
      0 1 2 3 4 5
    0 * * . . . .
    1 * * . . . .
    2 . . . . . .
    3 . * - - * .
    4 . - - - - .
    5 . * - - * .
    
    """
    img = np.zeros((6, 6, 3), dtype='uint8')
    img[0:3, 0:3] = 0
    img[0:3, 3:6] = 1
    img[3:6, 0:3] = 2
    img[3:6, 3:6] = 3
    
    annot = Annotation(
        categories=['0'],
        images={
            'img': AnnotatedImage(
                width=6,
                height=6,
                annotations=[AnnotatedObject(bbox=[0, 0, 2, 2], segmentation=[[0, 0, 1, 0, 1, 1, 0, 1]]),
                             AnnotatedObject(bbox=[1, 3, 4, 3], segmentation=[[1, 3, 4, 3, 4, 5, 1, 5]])]
            )
        }
    )
    
    img_path = os.path.join(os.path.dirname(__file__), 'test_files', 'img.png')
    cv2.imwrite(img_path, img)
    img_sources = paths2image_sources([img_path])
    dataset = Dataset(img_sources, annot)
    
    cropped_dataset = crop_dataset(dataset, (3, 3))
    cropped_dataset.split_by_proportions({'all': 1.0})
    dataset_path = os.path.join(os.path.dirname(__file__), 'test_files', 'cropped_dataset')
    cropped_dataset.install(
        dataset_path,
        image_ext='.png',
        install_images=True,
        install_yolo_det_labels = False,
        install_yolo_seg_labels = True,
      
    )
    
    # Check common properties
    images_dir = os.path.join(dataset_path, 'all', 'images')
    labels_dir = os.path.join(dataset_path, 'all', 'labels')
    assert len(os.listdir(images_dir)) == 4
    assert len(os.listdir(labels_dir)) == 4
    
    # Check images
    assert (cv2.imread(os.path.join(images_dir, 'img_0.png')) == 0).min() == True
    assert (cv2.imread(os.path.join(images_dir, 'img_1.png')) == 1).min() == True
    assert (cv2.imread(os.path.join(images_dir, 'img_2.png')) == 2).min() == True
    assert (cv2.imread(os.path.join(images_dir, 'img_3.png')) == 3).min() == True
    
    # Check labels   
    annot = read_yolo(labels_dir, (3, 3), ['0'])
    assert set(annot.images) == set(['img_0', 'img_1', 'img_2', 'img_3'])
    assert annot.images['img_0'].annotations[0].segmentation == [[0., 0., 0., 1., 1., 1., 1., 0.]]
    assert len(annot.images['img_1'].annotations) == 0
    assert annot.images['img_2'].annotations[0].segmentation == [[1., 0., 1., 2., 2., 2., 2., 0.]]
    assert annot.images['img_3'].annotations[0].segmentation == [[0., 0., 0., 2., 1., 2., 1., 0.]]
    
    os.remove(img_path)
    shutil.rmtree(dataset_path)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))