import sys
import pytest
import numpy as np
from shapely import Polygon, MultiPolygon
from datasculptor.tools import add_offset_to_objs, get_obj_union, join_splitted_annotations
from datasculptor import Annotation, AnnotatedImage, AnnotatedObject, Dataset, ImageSource


def test_add_offset_1():
    objs = [AnnotatedObject([1, 2, 3, 4])]
    new_objs = add_offset_to_objs(objs, 1, 2)

    assert new_objs[0].bbox == [2, 4, 3, 4]
    assert new_objs[0].segmentation == []


def test_add_offset_2():
    objs = [AnnotatedObject([1, 2, 3, 4], segmentation=[[1, 2, 3, 4, 1, 4]])]
    new_objs = add_offset_to_objs(objs, 1, 2)

    assert new_objs[0].bbox == [2, 4, 3, 4]
    assert new_objs[0].segmentation == [[2, 4, 4, 6, 2, 6]]


def test_get_obj_union_1():
    obj1 = AnnotatedObject([1, 1, 2, 2])
    obj2 = AnnotatedObject([1, 1, 2, 2])

    union_obj = get_obj_union(obj1, obj2)
    assert union_obj is None


def test_get_obj_union_2():
    bbox = [1, 1, 2, 2]
    segmentation = [[1, 1, 2, 2, 1, 2]]
    obj1 = AnnotatedObject(bbox, 0, segmentation)
    obj2 = AnnotatedObject(bbox, 1, segmentation)

    union_obj = get_obj_union(obj1, obj2)
    assert union_obj is None


def test_get_obj_union_3():
    bbox = [1, 1, 1, 1]
    segmentation = [[1, 1, 2, 2, 1, 2]]
    obj1 = AnnotatedObject(bbox, 0, segmentation)
    obj2 = AnnotatedObject(bbox, 0, segmentation)

    union_obj = get_obj_union(obj1, obj2)
    
    expected_poly = Polygon(np.array(segmentation[0]).reshape(-1, 2))
    got_poly = Polygon(np.array(union_obj.segmentation[0]).reshape(-1, 2))

    assert expected_poly.difference(got_poly).area < 1e-6
    assert pytest.approx(union_obj.bbox) == bbox
    assert union_obj.category_id == 0


def test_get_obj_union_4():
    bbox1 = [0, 0, 2, 2]
    bbox2 = [1, 0, 2, 2]
    segmentation1 = [[0, 0, 0, 2, 2, 2, 2, 0]]
    segmentation2 = [[1, 0, 1, 2, 3, 2, 3, 0]]

    obj1 = AnnotatedObject(bbox1, 0, segmentation1)
    obj2 = AnnotatedObject(bbox2, 0, segmentation2)

    union_obj = get_obj_union(obj1, obj2)
    
    expected_poly = Polygon([[0, 0], [0, 2], [3, 2], [3, 0]])
    got_poly = Polygon(np.array(union_obj.segmentation[0]).reshape(-1, 2))

    assert expected_poly.difference(got_poly).area < 1e-6
    assert pytest.approx(union_obj.bbox) == [0, 0, 3, 2]
    assert union_obj.category_id == 0


def test_get_obj_union_5():
    bbox1 = [0, 0, 1, 2]
    bbox2 = [1, 0, 2, 2]
    segmentation1 = [[0, 0, 0, 2, 1, 2, 1, 0]]
    segmentation2 = [[1, 0, 1, 2, 3, 2, 3, 0]]

    obj1 = AnnotatedObject(bbox1, 0, segmentation1)
    obj2 = AnnotatedObject(bbox2, 0, segmentation2)

    union_obj = get_obj_union(obj1, obj2)
    
    expected_poly = Polygon([[0, 0], [0, 2], [3, 2], [3, 0]])
    got_poly = Polygon(np.array(union_obj.segmentation[0]).reshape(-1, 2))

    assert expected_poly.difference(got_poly).area < 1e-6
    assert pytest.approx(union_obj.bbox) == [0, 0, 3, 2]
    assert union_obj.category_id == 0


def test_join_splitted_annotations_1():
    annot_img_0_0 = AnnotatedImage(
        width=5,
        height=5,
        annotations=[AnnotatedObject(segmentation=[[4, 0, 5, 0, 5, 1, 4, 1]])],
    )
    annot_img_0_1 = AnnotatedImage(
        width=5,
        height=5,
        annotations=[AnnotatedObject(segmentation=[[0, 0, 1, 0, 1, 1, 1, 1]])],
    )
    annot_img_1_0 = AnnotatedImage(
        width=5,
        height=5,
        annotations=[],
    )
    annot_img_1_1 = AnnotatedImage(
        width=5,
        height=5,
        annotations=[],
    )

    annot_images = [
        [annot_img_0_0, annot_img_0_1],
        [annot_img_1_0, annot_img_1_1],
    ]

    whole_annot_image = join_splitted_annotations(annot_images, (10, 10), 0)

    expected_poly = Polygon([[4, 0], [6, 0], [6, 1], [4, 1]])
    got_segm = whole_annot_image.annotations[0].segmentation
    got_poly = MultiPolygon([Polygon(np.array(s).reshape(-1, 2)) for s in got_segm])

    assert whole_annot_image.width == 10
    assert whole_annot_image.height == 10
    assert expected_poly.difference(got_poly).area < 1



if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))