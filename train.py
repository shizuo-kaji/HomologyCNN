# -*- coding: utf-8 -*-
#!/usr/bin/env python
#
# @brief homology assisted CNN for images
# @section Requirements:  python3,  chainer
# @version 0.01
# @date Aug. 2017
# @ author Shizuo KAJI

from __future__ import print_function
import argparse
import random
import sys,os
from pathlib import Path
from datetime import datetime as dt

import matplotlib as mpl
mpl.use('Agg')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from PIL import Image

import chainer
from chainer import training
from chainer.training import extensions,updaters
import chainer.functions as F

from chainerui.extensions import CommandsExtension
from chainerui.utils import save_args
from chainer.dataset import convert

import nin
import resnet
import functools

from dataset import comp_mean
from dataset import PreprocessedDataset as Dataset
#from dataset import DatasetFromDisk as Dataset           ### slower but requires less memory. use this for large dataset
from predict import predict_single

## global dictionary
archs = {
    'nin': nin.NIN,
    'ninres': nin.NINres,
    'resnet50': resnet.ResNet,
}

optim = {
    'Momentum': functools.partial(chainer.optimizers.MomentumSGD, lr=0.01, momentum=0.9),
    'AdaDelta': functools.partial(chainer.optimizers.AdaDelta,rho=0.95, eps=1e-06),
    'AdaGrad': functools.partial(chainer.optimizers.AdaGrad,lr=0.001, eps=1e-08),
    'Adam': functools.partial(chainer.optimizers.Adam,alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08),
    'RMSprop': functools.partial(chainer.optimizers.RMSprop,lr=0.01, alpha=0.99, eps=1e-08),
    'NesterovAG': functools.partial(chainer.optimizers.NesterovAG,lr=0.01, momentum=0.9)
}

activ = {
    'relu': F.relu,
    'leaky_relu': lambda x: F.leaky_relu(x, slope=0.02),
    'elu': F.elu,
    'tanh': F.tanh,
    'sigmoid': F.sigmoid
}

dtypes = {
    'fp16': np.float16,
    'fp32': np.float32
}

parser = argparse.ArgumentParser(description='homology assisted DNN for image classification')
parser.add_argument('--train', '-t', default='datatxt/kth-abc.txt', help='Path to image list file')
parser.add_argument('--val',  default='datatxt/kth-d.txt',
                    help='Validation data list. If set, everything in files will be used for training')
parser.add_argument('--train_ratio', '-tr', type=float, default=0.8,
                    help='when val is not specified, the train data will be split')
parser.add_argument('--gpu', '-g', type=int, nargs="*", default=None,
                    help='GPU IDs (negative value indicates CPU')
parser.add_argument('--initmodel', '-i',
                    help='Initialize the model from given file')
parser.add_argument('--loaderjob', '-j', type=int, default=3,
                    help='Number of parallel data loading processes')
parser.add_argument('--out', '-o', default='result',
                    help='Output directory')
parser.add_argument('--root', '-R', default='./texture/',
                    help='Root directory path of image files')
parser.add_argument('--snapinterval', '-si', type=int, default=50,
                    help='snapshot interval by epoch')
parser.add_argument('--predict', '-p', help='perform prediction using the specified model')
parser.add_argument('--regress', '-r', action='store_true', help='set for regression, otherwise classification')
parser.add_argument('--compute_mean', action='store_true')
parser.add_argument('--verbose', '-v', action='store_true')
parser.add_argument('--early_stopping', action='store_true')
parser.add_argument('--no-check', action='store_true', default=False, help='without type check of variables')
## training params
parser.add_argument('--batchsize', '-b', type=int, default=10,
                    help='minibatch size (reduce here for low memory)')
parser.add_argument('--val_batchsize', '-vb', type=int, default=10,
                    help='Validation minibatch size')
parser.add_argument('--epoch', '-e', type=int, default=200,
                    help='Number of epochs to train')
parser.add_argument('--optimizer', '-op', choices=optim.keys(), default='Momentum',
                    help='optimizer')
parser.add_argument('--weight_decay', '-w', type=float, default=1e-6,
                    help='weight decay for regularization')
parser.add_argument('--wd_norm', '-wn', choices=['l1','l2'], default='l1',
                    help='norm of weight decay for regularization')
## network params
parser.add_argument('--arch', '-a', choices=archs.keys(), default='nin',help='Convnet architecture')
parser.add_argument('--activation', '-na', choices=activ.keys(), default='relu',
                    help='Activation function')
parser.add_argument('--chs', type=int, nargs="*", default=[32,64,128,256],
                    help='number of channels in layers for NIN')
parser.add_argument('--fksize', '-kf', type=int, default=3,
                    help='size of the first convolution kernel')                        
parser.add_argument('--ksize', '-ks', type=int, default=3,
                    help='size of the second (and after) convolution kernel')                        
parser.add_argument('--n_resblock', '-nr', type=int, default=0,
                    help='number of residual blocks for NIN')                        
parser.add_argument('--conv_stride', '-cs', type=int, default=1,
                    help='stride for conv layer (1 for maxpooling)')                        
parser.add_argument('--drop_ratio', '-dr', type=int, default=0.4,
                    help='ratio of dropout')                        
parser.add_argument('--dtype', '-dt', choices=dtypes.keys(), default='fp32',
                    help='float precision')
parser.add_argument('--noise', type=float, default=0,
                    help='Noise injection')
### dataset params
parser.add_argument('--dataaug', '-da', type=int, default=3,
                    help='data augmentation by flip and random crop')                        
parser.add_argument('--mean', '-m',
                    help='Mean file (computed by --compute_mean)')
parser.add_argument('--hint_imgs', '-hi', nargs="*", default=[],
                    help='prefix for hint images')
parser.add_argument('--num_images', '-ni', type=int, default=1,
                    help='Number of images')
parser.add_argument('--num_class', '-nc', type=int, default=11,
                    help='Number of classes')
parser.add_argument('--size', '-sz', type=int, default=196,
                    help='crop size')

args = parser.parse_args()

#####################
def main():
    # create output directry 
    args.out = os.path.join(args.out, dt.now().strftime('%m%d_%H%M'))
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    print(args)

    # Enable autotuner of cuDNN
    chainer.config.autotune = True

    # Initialise the model
    print("\n\n Initialising...")
    if not args.gpu:
        if chainer.cuda.available:
            args.gpu = [0]
        else:
            args.gpu = [-1]            
    if len(args.gpu)==1 and args.gpu[0] >= 0:
        chainer.cuda.get_device_from_id(args.gpu[0]).use()
    
    if args.regress:
        accfun = F.mean_absolute_error
        lossfun = F.mean_squared_error
    else:
        accfun = F.accuracy
        lossfun = F.softmax_cross_entropy
        
    model = chainer.links.Classifier(archs[args.arch](args.num_class,
            params={
                'activation': activ[args.activation],
                'noise': args.noise,
                'channels': args.chs,
                'dtype':dtypes[args.dtype],
                'drop_ratio':args.drop_ratio,
                'n_resblock':args.n_resblock,
                'ksize':args.ksize,
                'first_ksize':args.fksize,
                'stride':args.conv_stride
            }), 
            accfun=accfun, lossfun=lossfun
        )

    if args.initmodel:
        print('Load model from: ', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)
    if args.predict:
        print('Load model from: ', args.predict)
        chainer.serializers.load_npz(args.predict, model)

    if len(args.gpu)==1 and args.gpu[0] >= 0:
        model.to_gpu()
    if args.no_check:
        chainer.config.type_check = False
    print('cuDNN {}'.format(chainer.cuda.cudnn_enabled))

    ## load dataset
    print("Load images from: ", args.root)
    with open(args.train) as input:
        filelist = input.readlines()
    
    ## if args.val is set, use it for validation
    if args.val:
        train_list = filelist
        with open(args.val) as input:
            val_list = input.readlines()
    else:  ## otherwise, split training dataset into train-val
        seed = None
        order = np.random.RandomState(seed).permutation(len(filelist))
        n_train = int(args.train_ratio*len(order))
        train_list = [filelist[i] for i in order[:n_train]]
        val_list = [filelist[i] for i in order[n_train:]]
        with open(args.out+"/train.txt",'w') as output:
            output.writelines(train_list)
        with open(args.out+"/val.txt",'w') as output:
            output.writelines(val_list)

    ## mean
    if args.mean:
        if args.mean.isdigit():
            mean = 0
        else:
            mean = np.load(args.mean)
    else:
        mean = None

    val = Dataset(val_list, args.root, mean, args.size, args.num_images, args.hint_imgs, args.regress, random=False, verbose=args.verbose)
    if args.compute_mean:
        train = Dataset(filelist, args.root, 0, args.size, args.num_images, [], random=False)
        comp_mean(args,train)
        exit()
    elif not args.predict:
        train = Dataset(train_list, args.root, mean, args.size, args.num_images, args.hint_imgs, args.regress, random=args.dataaug, verbose=args.verbose)
        val_iter = chainer.iterators.MultiprocessIterator(val, args.val_batchsize, repeat=False, shuffle=False, n_processes=args.loaderjob)
        # multi GPU with NCCL
        if len(args.gpu)>1:
            train_iters = [chainer.iterators.SerialIterator(i,args.batchsize)
                for i in chainer.datasets.split_dataset_n_random(train, len(args.gpu))]
        else:
            train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize, n_processes=args.loaderjob)

        # Set up an optimizer
        optimizer = optim[args.optimizer]()
        optimizer.setup(model)
        if args.weight_decay>0:
            if args.wd_norm =='l2':
                optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
            else:
                optimizer.add_hook(chainer.optimizer_hooks.Lasso(args.weight_decay))

        # Set up a trainer
        if len(args.gpu)>1:
            updater = updaters.MultiprocessParallelUpdater(train_iters,optimizer,devices=tuple(args.gpu))
        else:
            updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu[0])
        if args.early_stopping:
            stop_trigger = training.triggers.EarlyStoppingTrigger(verbose=True,max_trigger=(args.epoch, 'epoch'))
        else:
            stop_trigger = (args.epoch, 'epoch')
        trainer = training.Trainer(updater, stop_trigger, args.out)

        val_interval = args.snapinterval/10, 'epoch'
        snap_interval = args.snapinterval, 'epoch'
        log_interval = 200, 'iteration'
        print('intervals (epochs) -- snapshot: {}, valuation: {}'.format(args.snapinterval, args.snapinterval/10))

        trainer.extend(extensions.Evaluator(val_iter, model, device=args.gpu[0]),trigger=val_interval)
#        trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}'), trigger=snap_interval)
        trainer.extend(extensions.snapshot_object(
            model, 'model_epoch_{.updater.epoch}'), trigger=snap_interval)
        if args.epoch % args.snapinterval >0:
            trainer.extend(extensions.snapshot_object(
                model, 'model_epoch_{.updater.epoch}'), trigger=(args.epoch,'epoch'))
        trainer.extend(extensions.ProgressBar(update_interval=10))
        if args.optimizer in ['Momentum','AdaGrad','RMSprop']:
            trainer.extend(extensions.observe_lr(), trigger=log_interval)
            trainer.extend(extensions.ExponentialShift('lr', 0.5), trigger=(args.epoch/5, 'epoch'))
        elif args.optimizer in ['Adam']:
            trainer.extend(extensions.observe_lr(), trigger=log_interval)
            trainer.extend(extensions.ExponentialShift("alpha", 0.5, optimizer=optimizer), trigger=(args.epoch/5, 'epoch'))

        # log
        if args.verbose:
            trainer.extend(extensions.dump_graph('main/loss'))
        if extensions.PlotReport.available():
            trainer.extend(
                extensions.PlotReport(['main/loss', 'validation/main/loss','main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'))
        trainer.extend(extensions.PrintReport([
                'epoch', 'main/loss', 'main/accuracy', 'validation/main/loss','validation/main/accuracy', 'elapsed_time', 'lr'
            ]),trigger=log_interval)

        # resume from snapshot
#        if args.resume:
#            print('Resume from: ', args.resume)
#            chainer.serializers.load_npz(args.resume, trainer)

        # ChainerUI
        trainer.extend(CommandsExtension())
        save_args(args, args.out)
        trainer.extend(extensions.LogReport(trigger=log_interval))

        # start the trainer
        trainer.run()

    # prediction
    predict_single(args,val,model)

if __name__ == '__main__':
    main()
