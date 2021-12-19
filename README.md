

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

