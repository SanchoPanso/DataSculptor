# DataSculptor
## Setup

```
git clone https://github.com/SanchoPanso/DataSculptor.git
cd DataSculptor
python -m venv venv
source venv/bin/activate
pip install -e .
```

## GetStarted

Чтобы создать датасет, нужны данные и разметка. Начнем с разметки. 
Ее репрезентацией в пакете является класс Annotation.
```py
from datasculptor import Annotation

annot = Annotation()
```
Формат представления разметки имеет сходства с форматом COCO.
Аннотацию можно разделить на список категорий и размеченные изображения (класс AnnotatedImage).
```py
from datasculptor import Annotation, AnnotatedImage

annot = Annotation(categories=['a', 'b'], images={'1': AnnotatedImage()})
```

В свою очередь, каждое размеченное изображение имеет свой размер и свой набор размеченных объектов
(класс AnnotatedObject).

```py
from datasculptor import AnnotatedImage, AnnotatedObject

annot_img = AnnotatedImage(width=100, height=100, 
                           annotations=[AnnotatedObject(category_id=0, bbox=[10, 10, 20, 20])])
```

Разметку также можно прочитать и сохранить:
```py
from datasculptor import read_coco, write_coco, read_yolo, write_coco

annot = read_coco('coco.json')
write_coco(annot, 'coco.json')

annot = read_yolo('labels/')
write_coco(annot, 'labels/')

```

Помимо разметки, нужны данные, то есть исходные изображения. За их чтение, запись и обработку
отвечает базовый класс ImageSource. Его наследник PathImageSource отвечает за получение изображения
по заданному пути.

## Examples

Примеры создания датасетов показаны в папке examples

## Tests

```
python -m pytest tests
```


