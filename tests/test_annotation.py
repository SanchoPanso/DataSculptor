import pytest
import os
import sys
from pathlib import Path
import json

from datasculptor import Annotation, AnnotatedImage, AnnotatedObject, read_coco, write_coco, read_yolo, write_yolo_det

def test_init():
    annot = Annotation()
    
    # Check correctness of default initialization
    assert annot.categories == []
    assert annot.images == {}

    annot = Annotation(categories=['a', 'b'], images={})
    
    assert annot.categories == ['a', 'b']
    assert annot.images == {}

    
def test_read_coco_empty():
    coco_path = os.path.join(os.path.dirname(__file__), 'test_files', 'read_coco_empty.json')
    
    # Can be read as Path
    annot_1 = read_coco(Path(coco_path))
    
    # Can be read as str
    annot_2 = read_coco(coco_path)
    
    assert annot_1 == annot_2
    
    # Empty annotation was created correctly
    assert annot_2.categories == []
    assert annot_2.images == {}
    

def test_read_coco_default():
    coco_path = os.path.join(os.path.dirname(__file__), 'test_files', 'read_coco_default.json')
    annot = read_coco(coco_path)
    labeled_images = annot.images
    
    assert annot.categories == ['comet', 'other']
    assert set(labeled_images.keys()) == set(('1', '10'))

    assert len(labeled_images['1'].annotations) == 1
    assert len(labeled_images['10'].annotations) == 1
    
    assert labeled_images['1'].width == 2448  
    assert labeled_images['1'].height == 2048  
    
    image_1_annotations = labeled_images['1'].annotations
    assert image_1_annotations[0].category_id == 0
    assert image_1_annotations[0].segmentation == [[1200, 500, 1260, 500, 1200, 1050]]
    assert image_1_annotations[0].bbox == [1200, 500, 60, 550]
    
    assert labeled_images['10'].width == 3000  
    assert labeled_images['10'].height == 4000  
    
    image_2_annotations = labeled_images['10'].annotations
    assert image_2_annotations[0].category_id == 1
    # TODO for rle # assert image_2_annotations[0]['segmentation'] == {"size": [3000, 4000], "counts": [0, 1]}
    assert image_2_annotations[0].bbox == [560, 820, 60, 130]

def test_write_coco():

    annot = Annotation(categories=['comet', 'other'], 
                       images={
                           '1': AnnotatedImage(
                               width=2448, 
                               height=2048, 
                               annotations=[AnnotatedObject(
                                   category_id=0, 
                                   bbox=[1200, 500, 60, 550],
                                   segmentation=[[1200, 500, 1260, 500, 1200, 1050]])
                            ]
                            ),
                           
                            '10': AnnotatedImage(
                               width=3000, 
                               height=4000, 
                               annotations=[AnnotatedObject(
                                   category_id=1, 
                                   bbox=[560, 820, 60, 130])
                            ]
                            ),
                        })
    result_path = os.path.join(os.path.dirname(__file__), 'test_files', 'test_coco_2.json')
    write_coco(annot, result_path, image_ext='.png')

    coco_path = os.path.join(os.path.dirname(__file__), 'test_files', 'test_coco.json')
    with open(result_path) as f:
        got_dict = json.load(f)
    with open(coco_path) as f:
        expected_dict = json.load(f)
    
    assert got_dict['licenses'] == expected_dict['licenses']
    assert got_dict['info'] == expected_dict['info']
    print(got_dict['images'])
    print(expected_dict['images'])
    assert got_dict['images'] == expected_dict['images']
    assert got_dict['annotations'] == expected_dict['annotations']


def test_read_yolo():
    yolo_path = os.path.join(os.path.dirname(__file__), 'test_files', 'test_yolo')
    img_size = (2448, 2048)
    classes = ['comet', 'other']
    annot = read_yolo(yolo_path, img_size, classes)
    images = annot.images

    assert annot.categories == classes
    assert set(images.keys()) == set(('1', '10'))

    assert len(images['1'].annotations) == 1
    assert len(images['10'].annotations) == 1
    
    assert images['1'].annotations[0].category_id == 0
    assert list(map(int, images['1'].annotations[0].bbox)) == [1200, 500, 60, 550]
    
    
    assert images['10'].annotations[0].category_id == 1
    assert list(map(int, images['10'].annotations[0].bbox)) == [560, 820, 60, 130]
    

def test_write_yolo_det():

    classes = ['comet', 'other']
    images = {'1':  AnnotatedImage(2448, 2048, [AnnotatedObject([1200, 500, 60, 550], 0)]), 
              '10': AnnotatedImage(2448, 2048, [AnnotatedObject([560, 820, 60, 130],  1)])}
    annot = Annotation(classes, images)
    
    result_path = os.path.join(os.path.dirname(__file__), 'test_files', 'test_yolo_2')
    os.makedirs(result_path, exist_ok=True)
    write_yolo_det(annot, result_path)

    gt_path = os.path.join(os.path.dirname(__file__), 'test_files', 'test_yolo')
    
    for file in os.listdir(result_path):
        with open(os.path.join(result_path, file)) as f:
            got_str = f.read().strip()
        with open(os.path.join(gt_path, file)) as f:
            expected_str = f.read().strip()
        assert got_str == expected_str


def test_annotation_add():
    annot1 = Annotation(categories=['a', 'b', 'c'], 
                        images={
                            '1_1': AnnotatedImage(annotations=[AnnotatedObject(category_id=0)]),
                            '1_2': AnnotatedImage(annotations=[AnnotatedObject(category_id=1)]),
                            '1_3': AnnotatedImage(annotations=[AnnotatedObject(category_id=2)]),
                        })
    
    assert annot1 + Annotation() == annot1
    assert Annotation() + annot1 == annot1
    
    annot2 = Annotation(categories=['a', 'b', 'd'], 
                        images={
                            '2_1': AnnotatedImage(annotations=[AnnotatedObject(category_id=2)]),
                            '2_2': AnnotatedImage(annotations=[AnnotatedObject(category_id=0)]),
                            '1_3': AnnotatedImage(annotations=[AnnotatedObject(category_id=0)]),
                        })

    sum_annot = annot1 + annot2
    assert sum_annot.categories == ['a', 'b', 'c', 'd']
    assert sum_annot.images['1_1'].annotations[0].category_id == 0
    assert sum_annot.images['1_2'].annotations[0].category_id == 1
    
    assert sum_annot.images['2_1'].annotations[0].category_id == 3
    assert sum_annot.images['2_2'].annotations[0].category_id == 0
    
    assert sum_annot.images['1_3'].annotations[0].category_id == 2
    assert sum_annot.images['1_3'].annotations[1].category_id == 0
    
    

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
