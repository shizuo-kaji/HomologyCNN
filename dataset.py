# -*- coding: utf-8 -*-

from PIL import Image
import six
import numpy as np
import random
import sys
import os

import chainer
import chainer.datasets
from chainer.dataset import dataset_mixin

from chainercv.transforms import random_crop,center_crop
from chainercv.transforms import random_flip
from chainercv.transforms import resize
from chainercv.utils import read_image

# compute mean image
def comp_mean(args,train):
    print('compute mean image')
    sum_image = 0
    N = train.__len__()
    for i,z in enumerate(train):
        sum_image += z[0]
        sys.stderr.write('{} / {}\r'.format(i, N))
        sys.stderr.flush()
        sys.stderr.write('\n')
    print(sum_image.shape)
    sum_image /= N
    np.save(args.mean, sum_image)
    if sum_image.shape[0] == 3:
        Image.fromarray(np.uint8( (sum_image.transpose(1,2,0)+1)*127.5  )).convert('RGB').save("mean.png")
    elif sum_image.shape[0] == 1:
        Image.fromarray(np.uint8( (sum_image.transpose(1,2,0)+1)*127.5  )).convert('L').save("mean.png")

# MNIST
def save_mnist():
    train, test = chainer.datasets.get_mnist(ndim=2)
    for i in range(len(train)):
        t = train[i]
        Image.fromarray(np.uint8(t[0]*255)).save("{}_t{:05d}.jpg".format(t[1],i))
    for i in range(len(test)):
        t = test[i]
        Image.fromarray(np.uint8(t[0]*255)).save("{}_v{:05d}.jpg".format(t[1],i))
        

#############
class PreprocessedDataset(dataset_mixin.DatasetMixin):
    def __init__(self, filelist, datadir, mean, crop_size, num_im, hint_imgs, regress=False, random=False, verbose=False):
        self.crop_size = crop_size
        self.random = random
        self.filenames = []
        self.num_im = num_im     # use the first num_im images in the line
        self.images = []
        self.labels = []
        self.regress = regress
#        self.base = chainer.datasets.LabeledImageDataset(path, root)
        ## read the list of files
        for line in filelist:
            l = line.strip().split('\t')
            label = []
            filetxt = []
            for c in l:
                c=c.strip()                    
                if c.replace(".","").isdigit():    ## if float
                    if regress:
                        label.append(np.float32(c))
                    else:
                        label = np.int32(c)
                else:
                    filetxt.append(c)
            ## hint image filenames
            filetxt = filetxt[:num_im]
            for s in hint_imgs:
                for fn in filetxt[:num_im]:
                    fn1,ext = os.path.splitext(fn)
                    filetxt.append("{}_{}.png".format(fn1,s))
            if verbose:
                print(filetxt,label)
            ## stack images
            im_ch = []
            for i in range(len(filetxt)):
                col = True if i < num_im else False
                img = read_image(os.path.join(datadir,filetxt[i]),color=col)
                img = img * 2 / 255.0 - 1.0  # [-1, 1)
                c,h,w=img.shape
                # upscale if it is small
                if w<crop_size+random or h<crop_size+random:
                    ratio = max((crop_size+random)/w,(crop_size+random)/h)
                    img = resize(img, (int(w*ratio+0.1),int(h*ratio+0.1)))
                img = center_crop(img,(crop_size+random,crop_size+random))
                if i < num_im:
                    if mean:
                        img -= mean
                    else:
                        img -= np.mean(img)
                        img /= np.std(img)
                im_ch.append(img)
 
            image = np.concatenate(im_ch, axis=0)
            self.images.append(image)
            self.labels.append(label)
            self.filenames.append(filetxt)
 
        c,h,w = image.shape
        print("loaded: cropsize {} number {} channel {} height {} width {}".format(crop_size,len(self.images),c,h,w))
 
 
    def __len__(self):
        return len(self.images)
 
    def get_example(self, i):
        img = self.images[i]
        label = self.labels[i]
        if self.random:
            out = random_crop(img, (self.crop_size, self.crop_size))
            out = random_flip(out, x_random=True)
        else:
            out = center_crop(img, (self.crop_size, self.crop_size))
        return out, label

############# slower but reqires less memory
class DatasetFromDisk(dataset_mixin.DatasetMixin):
    def __init__(self, filelist, datadir, mean, crop_size, num_im, hint_imgs, regress=False, random=False, verbose=False):
        self.crop_size = crop_size
        self.random = random
        self.filenames = []
        self.num_im = num_im     # use the first num_im images in the line
        self.labels = []
        self.regress = regress
        self.datadir = datadir
        self.mean = mean
#        self.base = chainer.datasets.LabeledImageDataset(path, root)
        ## read the list of files
        for line in filelist:
            l = line.strip().split('\t')
            label = []
            filetxt = []
            for c in l:
                c=c.strip()                    
                if c.replace(".","").isdigit():    ## if float
                    if regress:
                        label.append(np.float32(c))
                    else:
                        label = np.int32(c)
                else:
                    filetxt.append(c)
            ## hint image filenames
            filetxt = filetxt[:num_im]
            for s in hint_imgs:
                for fn in filetxt[:num_im]:
                    fn1,ext = os.path.splitext(fn)
                    filetxt.append("{}_{}.png".format(fn1,s))
            if verbose:
                print(filetxt,label)
            self.labels.append(label)
            self.filenames.append(filetxt)

        print("loaded: cropsize {} number {} stack {}".format(crop_size,len(self.filenames),len(filetxt)))


    def __len__(self):
        return len(self.filenames)

    def get_example(self, i):
        ## load images from disk
        im_ch = []
        filetxt = self.filenames[i]
        crop_size = self.crop_size
        for j in range(len(filetxt)):
            img = read_image(os.path.join(self.datadir,filetxt[j]))
            img = img/127.5 - 1.0  # [-1, 1)
            c,h,w=img.shape
            # upscale if it is small
            if w<crop_size+self.random or h<crop_size+self.random:
                ratio = max((crop_size+self.random)/w,(crop_size+self.random)/h)
                img = resize(img, (int(w*ratio+0.1),int(h*ratio+0.1)))
            img = center_crop(img,(crop_size+self.random,crop_size+self.random))
            if j < self.num_im:
                if self.mean:
                    img -= self.mean
                else:
                    img -= np.mean(img)
                    img /= np.std(img)
            im_ch.append(img)
        img = np.concatenate(im_ch, axis=0)
        ## random data augmentation
        if self.random:
            out = random_crop(img, (self.crop_size, self.crop_size))   ## works for any number of channels
            out = random_flip(out, x_random=True)
        else:
            out = center_crop(img, (self.crop_size, self.crop_size))
        return out, self.labels[i]

if __name__ == '__main__':
    save_mnist()
