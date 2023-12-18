import sys
import logging

from datasculptor.annotation import Annotation, AnnotatedImage, AnnotatedObject
from datasculptor.annotation import read_coco, write_coco, write_yolo_det, write_yolo_iseg, read_yolo

from datasculptor.image_source import ImageSource, paths2image_sources, Resizer, Renamer
from datasculptor.det_dataset import DetectionDataset
from datasculptor.iseg_dataset import ISDataset
from datasculptor.dataset import Dataset
from datasculptor.dataset_cropping import crop_dataset

cvml_logger = logging.getLogger('datasculptor')

# Create handlers
s_handler = logging.StreamHandler(sys.stdout)
s_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
s_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
s_handler.setFormatter(s_format)

# Add handlers to the logger
cvml_logger.addHandler(s_handler)


