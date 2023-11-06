import sys
import logging

__version__ = '0.1.0'

from cvml2.annotation import Annotation, AnnotatedImage
from cvml2.annotation import read_coco, write_yolo_det, write_yolo_iseg

from cvml2.image_source import ImageSource, CropImageSource, paths2image_sources
from cvml2.det_dataset import DetectionDataset
from cvml2.iseg_dataset import ISDataset
from cvml2.dataset_cropping import crop_dataset

cvml_logger = logging.getLogger('cvml2')

# Create handlers
s_handler = logging.StreamHandler(sys.stdout)
s_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
s_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
s_handler.setFormatter(s_format)

# Add handlers to the logger
cvml_logger.addHandler(s_handler)


