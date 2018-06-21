#!/usr/bin/env python
# -*- coding: utf-8 -*-
# create file lists for cross validation

from __future__ import print_function
import argparse
import random

import numpy as np
import sys
import os

parser = argparse.ArgumentParser(description='Split image list file for cross-validation')
parser.add_argument('files', help='Path to image list file')
parser.add_argument('--n_folds', type=int, default=5, help='number of folds')
args = parser.parse_args()

with open(args.files) as input:
    filelist = input.readlines()

seed = None
order = np.random.RandomState(seed).permutation(len(filelist))
n_val = len(order)//args.n_folds

for i in range(args.n_folds):
    train_list = [filelist[i] for i in order[:i*n_val]]
    train_list.extend([filelist[i] for i in order[(i+1)*n_val:]])
    val_list = [filelist[i] for i in order[i*n_val:(i+1)*n_val]]
    train_list.sort()
    val_list.sort()
    with open("train{}.txt".format(i),'w') as output:
        output.writelines(train_list)
    with open("val{}.txt".format(i),'w') as output:
        output.writelines(val_list)

#for line in filelist:
#    l = line.strip().split('\t')
#    txt = l[1]+ "\tH_"+l[1]+ "\tH0_"+l[1] 
#    for i in range(len(l)):
#        c=l[i].strip()
#        if c.isdigit():
#            txt += "\t" + c

    