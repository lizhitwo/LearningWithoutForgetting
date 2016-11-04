# Learning without Forgetting

Created by [Zhizhong Li](http://zli115.web.engr.illinois.edu/) and [Derek Hoiem](http://dhoiem.cs.illinois.edu/) at University of Illinois, Urbana Champaign.

Project [webpage](http://zli115.web.engr.illinois.edu/learning-without-forgetting/).

Please contact [Zhizhong Li](http://zli115.web.engr.illinois.edu/) for any questions regarding this repository.

## Introduction

Learning without Forgetting aims at adding new capabilities (new tasks) to an existing Convolutional Neural Network, sharing representation with the original capabilities (old tasks), while allowing for adjusting the shared representation to adapt for both tasks without using the original training data.

The resulting network outperforms merely fine-tuning on the new tasks completely, suggesting it being a better practice than the widely-used method. It also outperforms feature extraction, but only on the new task performance. 

A more detailed abstract can be found in [our paper](https://arxiv.org/abs/1606.09282). The software aims at replicating our method. We use the [MatConvNet](http://www.vlfeat.org/matconvnet/) library.

## Usage

The code is tested on Linux (64 bit Arch Linux 4.4.5-1-ARCH)

### Prerequisites
0. Matlab (tested against R2015b)
0. MatConvNet v1.0-beta13
0. For GPU support, we use TITAN X with CUDA 7.5. 

### Installation
0. Compile MatConvNet accordingly using their installation guide.
0. Download the datasets you want to run experiments on, and place appropriately. (See Dataset Section)
0. Download [AlexNet/VGG](http://www.vlfeat.org/matconvnet/models/beta13/) pretrained on ImageNet from MatConvNet, preferrably from v1.0-beta13 model base. (See Dataset Section for placement)
0. Adjust ``getPath.m`` for your path setup for the datasets and models. (Usually just the ``p.path_dsetroot`` and ``p.path_exeroot`` value)
0. In matlab terminal, cd to the MatConvNet folder, run:

        run matlab/vl_setupnn
        addpath <path-to-LwF> % where you put <th></th>is repository
        addpath <path-to-LwF>/snippets % the subdirectories


### Experiments

Use ``gridSearchPASCAL(mode)`` as the entry point to this repository. Use different strings for ``mode`` to perform different experiments. See ``gridSearchPASCAL.m`` for details.

## Datasets and Models

The paper uses the following datasets. 

#### ImageNet
We use the [ILSVRC 2012](http://image-net.org/challenges/LSVRC/2012/signup) version. Place the folders ``ILSVRC2012_devkit_t12/`` and ``ILSVRC2012_img_*/`` under the ``<imgroot>/ILSVRC2012/`` folder. A few jpeg images has to be manually converted from CMYK to RGB.

The pre-trained models, ``imagenet-caffe-alex.mat`` and ``imagenet-vgg-verydeep-16.mat`` obtained from MatConvNet, are placed directly inside the ``<path-to-LwF>`` folder.

#### Places2
In this work, we use Places2 with 401 classes, which was used in the ILSVRC2015 taster challenge, so if you have the dataset available, you can set it up as follows.

Place the ``train/``, ``val/``, ``test/`` folders under ``<imgroot>/Places2/``. A ``train.txt`` and ``val.txt`` should also be placed here, with each line defining one image's relative path and the label:

    (train.txt)
    a/abbey/00000000.jpg 0
    a/abbey/00000001.jpg 0
    a/abbey/00000002.jpg 0
    a/abbey/00000003.jpg 0
    ...

    (val.txt)
    Places2_val_00000001.jpg 330
    Places2_val_00000002.jpg 186
    Places2_val_00000003.jpg 329
    Places2_val_00000004.jpg 212

Unfortunately, this Places2 version has become obsolete after our paper submission. [Places365](http://places2.csail.mit.edu/) now replaces it as a better dataset, but we have not yet adjusted our code accordingly.

The provided CNN once available from their website should be fine-tuned to the downscaled dataset due to stability issues, and converted to MatConvNet. We provide our fine-tuned conversion of [Places2 CNN](https://drive.google.com/file/d/0B9hb19EgsNunVFBUVE5SclFXbkE/view?usp=sharing). This should be placed in the ``<path-to-LwF>/convertcaffe/`` folder using filename ``alexnet_places2_iter_158165_convert.mat``.

#### PASCAL VOC

We use the [VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) version. Place the folders such as ``JPEGImages/`` and ``ImageSets/`` under the ``<imgroot>/VOC2012/`` folder.

#### MIT indoor scene

The [MIT indoor scene](http://web.mit.edu/torralba/www/indoor.html) dataset. Folder ``Images/`` and files ``trainImages.txt``, ``testImages.txt`` are to be placed under ``<imgroot>/MIT_67_indoor/``. NOTE: please rename the .txt files (no front caps).

A ``classes.txt`` should be generated using shell command line:

    cd Images
    ls -d * > ../classes.txt

within the folder.

Some of the images (see [MIT_wrongformat.txt](snippets/MIT_wrongformat.txt)) may be of erroneous format and should be saved as actual *.jpg format. You can use e.g. an image editing software to convert them.

#### Caltech-UCSD-Birds

We use the [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) version (the larger one). Place folders such as ``images/`` and files such as ``classes.txt`` into the ``<imgroot>/CUB_200_2011/`` folder.

Copy the provided ``snippets/convert_CUB_old.py`` script into the folder, and execute it in shell ``python convert_CUB_old`` to generate the train/test image lists. You need Python for this step.

#### MNIST

Please place the [four files provided by MNIST](http://yann.lecun.com/exdb/mnist/) into ``<imgroot>/MNIST/``, and unzip them.


## Notes
0. Due to our implementation, the efficiency for LwF here is actually similar to joint training instead of being better; theoretically it can be optimized by sharing fwd/bkwd pass of the shared layers across tasks using e.g. the dagnn for MatConvNet 0.17 upwards, or other libraries such as tensorflow.


## Citation

Please cite our paper if you use our work in your research.

    @inproceedings{li2016learning,
        title={Learning Without Forgetting},
        author={Li, Zhizhong and Hoiem, Derek},
        booktitle={European Conference on Computer Vision},
        pages={614--629},
        year={2016},
        organization={Springer}
    }

## Acknowledgements

The MNIST helper files are generously provided by [Stanford UFLDL](http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset).

Some of our code are based on MatConvNet code  and VOC devkit.


## License

This software package is freely available for research purposes. Please check the LICENSE file for details.