# Fruit detection using YOLOv5

The food we eat is receiving a lot of attention due to the fast development of technology.
Skilled labor is one of the most expensive components in the agricultural business.
The industry is leaning toward automation in order to reduce labor costs while improving quality.
Robotic harvesting has the ability to bring viable answers to many of the industry's challenges.
The forthcoming technology will have to complete a number of difficult tasks, one of which is an accurate fruit detecting system.
Various methods, including new computer vision technologies, have been employed in the past for fruit detection.

The specific study involves developing a robust model for fruit detection.
There might be a plethora of sophisticated applications for this.
Among them are:

1. You work in a warehouse where lakhs of fruits arrive everyday, and physically separating and packaging each fruit box will take a large amount of time.
As a result, you may create an automated system that detects fruits and separates them for packing.

2. You are the proud owner of a massive orchid.
Manual harvesting of the fruits will also need a large staff.
You may create a robot or a self-driving vehicle that can recognize and pick fruits from specified trees. 

## Aim

To build a robust fruit detection system using YOLOv5.
Tech stack
- Language: Python
- Object detection: YOLOv5
- Data annotation: Robotflow
- Environment: Google Colab

### Data collection and Labeling

To create a custom object detector, we need an excellent dataset of images and
labels so that the sensor can efficiently train to detect objects.
We can do this in two ways.

 #### a. Using Google's Open Images Dataset
We can gather thousands of images and their auto-generated labels within
minutes. [Explore that dataset here!](https://storage.googleapis.com/openimages/web/index.html)

 #### b. Creating your dataset and then labelling it manually
We will create a dataset manually by collecting images from google image
scraper or manually clicking them and then marking them using an image
annotation tool.

## YOLO v5 Model Architecture

 There is no paper on YOLOv5 as of August 1, 2021.
As a result, this essay will go through YOLOv4 in detail so that you can comprehend YOLOv5.
To further understand how Yolov5 enhanced speed and design, consider the following high-level Object detection architecture: 
![source](https://github.com/adrienpayong/object-detection/blob/main/Capture1.PNG).

A backbone will be used to pre-train the General Object Detector, and a head will be used to predict classes and bounding boxes.
The Backbones may operate on either GPU or CPU platforms.
For Dense prediction, the Head may be one-stage (e.g., YOLO, SSD, RetinaNet) or two-stage (e.g., Faster R-CNN) for the Sparse prediction object detector.
Recent Object Detectors contain certain layers (Neck) to gather feature maps, which are located between the backbone and the Head.

In YOLOv4, [CSPDarknet53](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w28/Wang_CSPNet_A_New_Backbone_That_Can_Enhance_Learning_Capability_of_CVPRW_2020_paper.pdf) is utilized as a backbone and [SPP](https://arxiv.org/abs/1406.4729) block to increase the receptive field, which isolates the essential features, and the network operation performance is not reduced.
PAN is used to aggregate parameters from multiple backbone levels.
For YOLOv4, the YOLOv3 (anchor-based) head is utilized. 
YOLOv4 included two new data augmentation methods: Mosaic and Self-Adversarial Training (SAT).
[Mosaic](https://arxiv.org/pdf/2004.10934.pdf) is a combination of four training images.
Self-Adversarial Training is divided into two stages: forward and backward.
In the first step, the network modifies merely the image rather than the weights.
The network is trained to detect an object on the changed image in the second step. 
Yolov5 is similar to Yolov4 but differs in the following ways:
- Yolov4 was provided as part of the Darknet framework, which is written in C.
Yolov5 is built on top of the PyTorch framework.
- Yolov4 configures using.cfg files, however Yolov5 configures using.yaml files. 

For further details, please see the [Yolov5 Github repository](https://github.com/ultralytics/yolov5). 

### Steps Covered in this project

To train our detector we take the following steps:

1) Install YOLOv5 dependencies

YOLOv5 is built by Ultralystic using the PyTorch framework, which is one of the most popular in the AI community.
However, this is simply a basic design; researchers may modify it to provide the best results for specific issues by adding layers, eliminating blocks, including new image processing techniques, modifying the optimization methods or activation functions, and so on. 

```
# clone YOLOv5 repository
!git clone https://github.com/ultralytics/yolov5  # clone repo
%cd yolov5
!git reset --hard 886f1c03d839575afecb059accf74296fad395b6
# install dependencies as necessary
!pip install -qr requirements.txt  # install dependencies (ignore errors)
import torch
from IPython.display import Image, clear_output  # to display images
from utils.google_utils import gdrive_download  # to download models/datasets
```
2) Download a Custom Dataset that is Properly Formatted

Roboflow will be used to download our dataset. The "YOLOv5 PyTorch" output format should be used.
It's worth noting that the Ultralytics solution requires a YAML file that specifies the location of your training and test data.
This format is also written for us by the Roboflow export. 
```
#follow the link below to get your download code from from Roboflow
!pip install -q roboflow
from roboflow import Roboflow
rf = Roboflow(model_format="yolov5", notebook="roboflow-yolov5")
```
3) Specify the Model Configuration and Architecture

We'll create a yaml script that specifies the parameters for our model, such as the number of classes, anchors, and layers.
The images are accessed and used as input by the YOLOv5 model on PyTorch through a yaml file providing summary information about the data set.
The format of the data.yaml file used in the YOLO model is as follows:
- train: ‘training set directory path’
- val: ‘validation set directory path’
- nc: ‘number of classes’
- names: ‘name of objects'


Because the specified dataset does not include a data.yaml file, it must be initialized.
Typically, this data.yaml file is created in Notepad or Notepad ++, then saved in yaml format and uploaded to Drive.
However, it will be written directly in Colab in this case.

```
# define number of classes based on YAML
import yaml
with open(dataset.location + "/data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])
```
To be able to overwrite the empty yaml file, a function from iPython.core.magic must be imported. 

```
#customize iPython writefile so we can write variables
from IPython.core.magic import register_line_cell_magic

@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))
```

Glenn Jocher additionally includes various sample YOLOv5 models based on prior theory.
On PyTorch, the YOLOv5 model will read these architectures from the yaml file and create them in the train.py file.
This also simplifies the configuration of the architecture based on the various object detection challenges. 
```
%%writetemplate /content/yolov5/models/custom_yolov5s.yaml
# parameters
nc: {num_classes}  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple

# anchors
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Focus, [64, 3]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, BottleneckCSP, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 9, BottleneckCSP, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, BottleneckCSP, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 1, SPP, [1024, [5, 9, 13]]],
   [-1, 3, BottleneckCSP, [1024, False]],  # 9
  ]

# YOLOv5 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, BottleneckCSP, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, BottleneckCSP, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, BottleneckCSP, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, BottleneckCSP, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```
