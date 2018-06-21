# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import sys
import os
from PIL import Image,ImageDraw
import shutil
import pandas as pd

import chainer
from chainer import training,datasets
from chainer.training import extensions
import chainer.functions as F
from chainer.training import updaters
from chainer.dataset import convert, concat_examples

## predictor for single images
def predict_single(args,val,model):
    xp = model.xp
#    converter=convert.ConcatWithAsyncTransfer()
    converter=concat_examples
    iterator = chainer.iterators.SerialIterator(val,args.val_batchsize,repeat=False, shuffle=False)
    print("\n\n Predicting...")
    output = ""
    ## for regression
    if args.regress:
        result = []
        col = ['name','truth','prediction']
        idx = 0
        for batch in iterator:
            x, t = converter(batch, device=args.gpu[0])
            with chainer.using_config('train', False):
                with chainer.function.no_backprop_mode():
                    y = model.predictor(x).data
                    if args.gpu[0]>-1:
                        y = xp.asnumpy(y)
            for i in range(len(y)):
                fn = val.filenames[idx][0]
                output += "{}, true {}, pred {}\n".format(fn,t[i],y[i])
                result.append(pd.DataFrame([[fn,t[i],y[i]]],columns=col))
                idx += 1
        output_info = ''
    ## for classification
    else:
        fail = 0
        idx = 0
        result = []
        col = ['name','truth','prediction']+[i for i in range(args.num_class)]
        for batch in iterator:
            x, t = converter(batch, device=args.gpu[0])
            with chainer.using_config('train', False):
                with chainer.function.no_backprop_mode():
                    y = F.softmax(model.predictor(x)).data
                    if args.gpu[0]>-1:
                        y = xp.asnumpy(y)
            for i in range(len(y)):
                pred = np.argmax(y[i])
                fn = val.filenames[idx][0]
                if pred != t[i]:
                    output += "{}, true {}, pred {}\n".format(fn,t[i],pred)
                    if(args.verbose):
                        shutil.copy(os.path.join(args.root,fn), os.path,join(args.out,"fail_",fn))
                    fail += 1
                result.append( pd.DataFrame([[fn,str(t[i]),pred]+y[i].tolist()],columns=col) )
                idx += 1                
        output_info = "Fail {}, Total {}, Accuracy {}\n".format(fail,len(val),1-float(fail)/len(val))

    output_info += "command line arguments \n"
    arg_dict = vars(args)
    for key in arg_dict:       
        output_info += '{}:{}\n'.format(key,arg_dict[key])
    print(output_info)

    output_info += "\n\n" + output
    # write results to file
    with open(os.path.join(args.out,"summary.txt"),'w') as out:
        out.writelines(output_info)

    result = pd.concat(result,ignore_index=True)
    result.to_csv(os.path.join(args.out,"result.csv"))
