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

## Approach
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


