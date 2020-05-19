Homology-assisted Convolutional Neural Network for 2D/3D image classification
=============
by Shizuo KAJI (skaji@imi.kyushu-u.ac.jp)

The image classification problem is to 
train a model to classify images into a predefined number of classes given a set of labelled images.
The problem is commonly solved by convolutional neural networks (CNNs).

In general, CNNs are better at learning local features than global features.
So it would increase the performance if we could _teach_ global information to CNNs.
We propose a kind of feature engineering, where the input images for a CNN
 are preprocessed by being supplemented with global information encoded by _persistent homology_,
 a mathematically defined and calculable image feature.

This _mariage_ of machine learning and mathematics generally performs better than CNN alone.
The technique is independent of network structure, and can be used straightforwardly in conjunction with existing systems.

The algorithm first computes "homology images" using _persistent homology_.
Persistent homology is a popular tool in _Topological Data Analysis_ (TDA), which captures the topology or the _shape_ of data.
A homology image is a multi-channel image of the same dimension as the input, 
with the generators of cycles drawn with the intensity according to their lifetime (or size). 

![homology image](https://github.com/shizuo-kaji/HomologyCNN/blob/master/homology.jpg?raw=true)

The homology image will be bundled (as an extra channel) with the original image and fed into a CNN for classification.

We demonstrate the idea with Reduced MNIST and 3D MNIST classification tasks.

## Licence
MIT Licence

## Requirements
- a modern GPU
- python 3: [Anaconda](https://anaconda.org) is recommended
- PyTorch, Keras, Tensorflow (2.1.0)

Preprocessed datasets are included, but if you want to create the dataset yourself, download
- (mnist.pkl.gz) https://github.com/mnielsen/rmnist/tree/master/data
- (full_dataset_vectors.h5) https://www.kaggle.com/daavoo/3d-mnist
- and install CubicalRipser by

    % pip install git+https://github.com/shizuo-kaji/CubicalRipser_3dim

## Experiments
You can control e.g., the network structure by giving command-line arguments.
Each script gives a brief list of command-line arguments when invoked with ''-h''. 

Reduced MNIST (n=1, only one sample from each class) without persistent homology (i.e., the ordinary way)
```
python mnist_torch.py -n 1 -c 0 -s 56
```

Reduced MNIST (n=1, only one sample from each class) with persistent homology (lifetime image stacked to the original image)
```
python mnist_torch.py -n 1 -s 56
```

3D MNIST without persistent homology
```
python mnist3d_keras.py -c 0 -n 10000
```
Note that volumes in 3D MNIST are applied arbitrary rotation.
So we use the whole data (n=10000) for training.
We might be able to do with a smaller n by applying data augmentation.

3D MNIST with persistent homology (lifetime image stacked to the original image)
```
python mnist3d_keras.py -n 10000
```

The results (accuracy, top-2 accuracy, per class accuracy) are shown below:

![Reduced MNIST (n=1) Result](https://github.com/shizuo-kaji/HomologyCNN/blob/master/rmnist_total.jpg?raw=true)
![Reduced MNIST (n=1) Per Class Result](https://github.com/shizuo-kaji/HomologyCNN/blob/master/rmnist_class.jpg?raw=true)

![3D MNIST Result](https://github.com/shizuo-kaji/HomologyCNN/blob/master/3dmnist_total.jpg?raw=true)
![3D MNIST Per Class Result](https://github.com/shizuo-kaji/HomologyCNN/blob/master/3dmnist_class.jpg?raw=true)

## Data preprocessing
You can play with different preprocessing using make_mnist.py.
Results are saved in ".npz" files.

Create Reduced MNIST (n=1) scaled to 56x56 stacked with lifetime image
```
python make_mnist.py -d 2 -n 1 -s 56 -t --ph life
```

Create 3D MNIST stacked with lifetime image
```
python make_mnist.py -d 3 -n 10000 -t --ph life
```


# Obsolete
This section describes an old version of this project.
We no longer maintain it. The files are found under "old" directory.

## Requirements
- Python 3: [Anaconda](https://www.anaconda.com/download/) is recommended
- Python libraries: Chainer, cupy chainercv, chainerui:  `pip install cupy chainer chainerui chainercv`
- R: [Microsoft R Open](https://mran.microsoft.com/open) is recommended
- R libraries: TDA,imager,ggplot2: `install.packages(c("ggplot2","TDA","imager"))` from R
- CUDA supported GPU is highly recommended. Without one, it takes ages to do anything with CNN.

# How to use
The usage will be demonstrated with a texture classification problem using the
[KTH-TIPS2-b](http://www.nada.kth.se/cvap/databases/kth-tips/index.html) dataset.

Download the KTH-TIPS2-b dataset and extract the archive.
Copy all png files into a single directory, say, named "dataset/texture".

First, we need one text file for each train/test dataset, containing lines with
```
    "ImageFileName   class"
    "ImageFileName   class"
    ...
```
I have included sample files for KTH-TIPS2-b under "datatxt".

Homology images should be computed in advance. We use R for this part.
This procedure takes a bit of time.
```
Rscript compute_PH.R dir png
```
produces persistent homology images from image files under "dir" with filename extension "png".
Note that "dir" should be the full path for the directory containing the images.
Homology images will be put under "dir" with some suffix like "_Hsup1_life" in the file names.
Modify the beginning of the R script "compute_PH.R" to tune parameters, if you wish.

Other tasks will be done by the python script. 
```
python train.py -h
```
gives a brief description of command line arguments.

A typical training is performed by
```
python train.py -t datatxt/kth-abc.txt --val datatxt/kth-d.txt -R dataset/texture -a nin -e 200 -op Adam --num_class 11 -hi Hsup1_life
```
Logs and learnt model files will be placed under "result" directory. 

You can also train a CNN without using homology.
```
python train.py -t datatxt/kth-abc.txt --val datatxt/kth-d.txt -R dataset/texture -a nin -e 200 -op Adam --num_class 11
```
Compare the performances.

Inference using a learnt model is done by
```
python train.py --val datatxt/kth-d.txt -R dataset/texture -a nin --num_class 11 -hi Hsup1_life -p model_epoch_100
```
"model_epoch_100" is the model file produced by training. 