import os
import sys
import pytest
import numpy as np
from pathlib import Path
from easydict import EasyDict as edict

# sys.path.append(str(Path(__file__).parent.parent.parent))
from datasculptor import Annotation, AnnotatedImage, AnnotatedObject, DetectionDataset
from datasculptor import paths2image_sources


def test_init():
    d1 = DetectionDataset()
    assert d1.image_sources == []
    assert d1.annotation == Annotation()
    assert d1.subsets == {}
    
    img_srcs = paths2image_sources(['1.png', '2.png', '3.png'])
    categories = ['category']
    images = {}
    annot = Annotation(categories=categories, images=images)
    d2 = DetectionDataset(img_srcs, annot)
    assert d2.image_sources == img_srcs
    assert d2.annotation == annot
    assert d2.subsets == {}


def test_rename():
    
    # Check empty dataset renaming
    d1 = DetectionDataset()
    d1.rename(lambda x: x + '_')
    assert d1.image_sources == []
    assert d1.annotation == Annotation()
    
    # Check default renaming
    img_paths = paths2image_sources(['1.png', '2.png'])
    categories = ['category']
    images = {'1': AnnotatedImage(), '2': AnnotatedImage()}
    annot = Annotation(categories=categories, images=images)
    d2 = DetectionDataset(img_paths, annot)
    d2.rename(lambda x: x + '_')
    assert [src.name for src in d2.image_sources] == ['1_', '2_']
    assert set(d2.annotation.images.keys()) == set(['1_', '2_'])


def test_resize():
    img_srcs = paths2image_sources(['1.png'])
    categories = ['category']
    images = {'1': AnnotatedImage(width=10, height=10, annotations=[AnnotatedObject(bbox=[1, 2, 3, 4])])}
    annot = Annotation(**{'categories': categories, 'images': images})
    d1 = DetectionDataset(img_srcs, annot)
    
    width, height = 20, 30
    d1.resize((width, height))
    
    img = np.zeros((1, 1), dtype='uint8')
    prep_fn = d1.image_sources[0].preprocessing_fns[0]
    prep_img = prep_fn(img)
    
    assert prep_img.shape == (height, width)
    assert d1.annotation.images['1'].width == width
    assert d1.annotation.images['1'].height == height
    assert d1.annotation.images['1'].annotations[0].bbox == [2, 6, 6, 12]
    

def test_magic_add():
    imsrc = ''
    d1 = DetectionDataset([imsrc, imsrc, imsrc], None, {'train': [0, 1], 'valid': [2], 'test': []})
    d2 = DetectionDataset([imsrc, imsrc, imsrc], None, {'train': [0], 'valid': [1, 2]})

    d3 = d1 + d2

    assert len(d3.image_sources) == 6
    assert d3.subsets['train'] == [0, 1, 3]
    assert d3.subsets['valid'] == [2, 4, 5]
    assert d3.subsets['test'] == []


# def test_add_with_proportions():
#     imsrc = ''

#     d1 = DetectionDataset([imsrc, imsrc, imsrc], None, {'train': [0, 1], 'valid': [2], 'test': []})
#     d2 = DetectionDataset([imsrc], None, {})

#     d3 = d1.add_with_proportion(d2, {'train': 0.5, 'valid': 0.4, 'test': 0.1})

#     assert len(d3.image_sources) == 4
#     assert d3.samples['train'] == [0, 1]
#     assert d3.samples['valid'] == [2, 3]
#     assert d3.samples['test'] == []
    
    
#     d4 = DetectionDataset([], None, {'train': [], 'valid': []})
#     d5 = DetectionDataset([imsrc], None, {})
    
#     for i in range(10):
#         d4 = d4.add_with_proportion(d5, {'train': 0.5, 'valid': 0.5})
    
#     assert len(d4.samples['train']) == 5
#     assert len(d4.samples['valid']) == 5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))