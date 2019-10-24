# Object detection of 3D-printed objects using synthetic images

This is the repository based on my bachelor thesis in computer since.<br/>
There is also a dataset (images, pretrained models etc.) available through one of the following links:
* https://bit.ly/2oBSpOz (Google Drive)
* https://tubcloud.tu-berlin.de/s/QwTo9aNtxMA588n (Nextcloud of TU Berlin)

## Repository

This repository generally contains different Python and Jupyter Notebook files
to process data and train convolutional neural networks (CNNs). It's written in Python 3.7
with the Keras-API with Tensorflow backend and OpenCV.
I have provided a `requirements.txt`. It is recommended to use a virtual environment.
In the following a short overview is given:
* `classification/` contains different CNNs for the classification task. One is based on the VGG-16 architecture, the other is my own architecture... there is also a notebook with the K-Nearest-Neighbor algorithm.
* `color_detection/` contains a detector for object dection of white printed 3D-objects on a black background. It crops the objects based on their color and classifies them with the CNNs declared in `classification/`.
* `data/` contains just a file with the classes of the test dataset. Obviously you should place the data here ;-)
* `faster_rcnn/` contains an implementation of the [Faster R-CNN](https://arxiv.org/abs/1506.01497) based on [this repository](https://github.com/RockyXu66/Faster_RCNN_for_Open_Images_Dataset_Keras) (but much faster, since numpy is used for parallelization).
* `preprocessing/` contains... huh. I'am sure you know what it contains ;-) Something with data I guess...
* `rendering/` contains different scripts to render the synthetic training data. It's based on [blender 2.79](https://www.blender.org).

The `test.py` files save the predictions of the models on disk.
To calculate the mean average precision (mAP) consider https://github.com/Cartucho/mAP.

## Dataset

The are different files that build the dataset:
* `3Dobjects.tar.gz` - 45 `.stl` files of the 3D-Objects of the dataset
* `frcnn_weights_trained64.tar.gz` - pretrained Faster R-CNN for the 45 3D-Objects
* `rendered_transp_512_16_8_1.tar.gz` - rendered images of the 3D-Objects in different positions and rotations
* `test_images.tar.gz` - annotated natural images for testing different approaches
* `tiny-yolo3D.tar.gz` and `yolo3D.tar.gz` - pretrained [YOLOv3](https://pjreddie.com/darknet/yolo/)

In the folder `cnn_classification/` you will find pretrained weights and Keras-Models
of different CNNs. There are also files like `featurewise_mean.npy` and `featurewise_std.npy`
for data normalization of the CNN-Models.

All weights are trained on Google Cloud Compute Engines with 4 vCPUs, 16GB RAM and a Tesla K80 GPU.
