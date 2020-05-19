import _pickle as cPickle
import gzip
import random
import numpy as np
import os,time,subprocess,glob
from stackPH import lifeVect,histVect
from scipy.ndimage.morphology import distance_transform_edt
from skimage.filters import threshold_otsu
from skimage.transform import resize
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import argparse
import h5py

# you need cubical ripser
import cripser

def make_rmnist(n=10,size=None,dim=3,save_test=False):
    if dim == 2:
        with gzip.open("mnist.pkl.gz", 'rb') as f: ## the original MNIST
            td, vd, ts = cPickle.load(f, encoding='latin1')
            train_x = np.array(td[0]).reshape((-1,28,28))
            train_y = np.array(td[1],dtype=np.uint8)
            test_x = np.array(ts[0]).reshape((-1,28,28))
            test_y = np.array(ts[1],dtype=np.uint8)
    elif dim == 3: ## 3D MNIST in Kaggle
        with h5py.File("full_dataset_vectors.h5", "r") as hf:
            train_x = np.array(hf["X_train"][:]).reshape(-1,16,16,16)
            train_y = np.array(hf["y_train"][:])
            test_x = np.array(hf["X_test"][:]).reshape(-1,16,16,16)
            test_y = np.array(hf["y_test"][:])
    if n<len(train_x):
        indices = np.random.permutation(len(train_y))
        sub_indices = np.concatenate([np.where(train_y==j)[0][:n] for j in range(10)])
        np.random.shuffle(sub_indices)
        train_x = train_x[sub_indices]
        train_y = train_y[sub_indices]
        print("Used indices:",sub_indices)

    if size != train_x[0].shape[0]:
        if dim == 2:
            out_shape = (size,size)
        else:
            out_shape = (size,size,size)
        train_x = np.stack([resize(train_x[j],out_shape) for j in range(len(train_x))],axis=0)
        if save_test:
            test_x = np.stack([resize(test_x[j],out_shape) for j in range(len(test_x))],axis=0)

    print("input shape:", train_x.shape,test_x.shape)
    return(train_x,train_y,test_x,test_y)

def plot_mnist(dat):
    fig = plt.figure(figsize=(10, 2))
    grid = ImageGrid(fig, 111,nrows_ncols=(1, len(dat)),axes_pad=0.1)
    for i, img in enumerate(dat[:,0]):
        bw_img = (img >= threshold_otsu(img))
        dat[i,0] = (distance_transform_edt(bw_img)-distance_transform_edt(~bw_img))/5+0.5
    if dat.shape[1]>3:
        dat[:,1] = np.sum(dat[:,1:((dat.shape[1]+1)//2)],axis=1)/2
        dat[:,2] = np.sum(dat[:,((dat.shape[1]+1)//2):],axis=1)/2
    for ax, im in zip(grid, dat[:,:3]):
        ax.imshow(im.transpose(1,2,0))
    print(np.max(dat[:,1],axis=(1,2)),"\n",np.max(dat,axis=(2,3)))        
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')
    parser.add_argument('--train_sample', '-n', default=1, type=int, help='number of train samples in each class')
    parser.add_argument('--size', '-s', default=0, type=int, help='size of image')
    parser.add_argument('--dim', '-d', default=2, type=int, choices=[2,3], help='dimension of image')
    parser.add_argument('--max_life', '-Ml', default=5, type=int, help='max life')
    parser.add_argument('--min_life','-ml', type=float, default=0, help="minimum lifetime to be counted")
    parser.add_argument('--ph', default="life", choices=["life","hist"], help="persistence encoding scheme")
    parser.add_argument('--plot', '-p', action='store_true',help="just plot")
    parser.add_argument('--save_test', '-t', action='store_true',help="create test dataset in addition to training dataset")
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    np.random.seed(0)

    n = args.train_sample
    if args.dim == 2:
        dim_str = ""
        if args.size == 0:
            args.size = 28
    else:
        dim_str = "3d"
        if args.size == 0:
            args.size = 16
    x_filename = ["rmnist{}{}_s{}_train_{}.npz".format(dim_str,n,args.size,args.ph),
                    "mnist{}_s{}_test_{}.npz".format(dim_str,args.size,args.ph)]

    if args.plot:
        k=4
        if args.save_test:
            dat = np.load(x_filename[1])
        else:
            dat = np.load(x_filename[0])

        print(dat['y'][:k])
        if args.dim==2:
            plot_mnist(dat['x'][:k])
        else:
            plot_mnist(dat['x'][:k,:3,:,:,dat['x'].shape[-1]//2])
        exit()

    tx,ty,vx,vy = make_rmnist(n,size=args.size,dim=args.dim,save_test=args.save_test)

    if args.dim==2:
        mx,my = tx[0].shape
        mz = 1
        bd = [0,args.size-1]
    else:
        mx,my,mz = tx[0].shape
        bd = [0,15]

    if args.save_test:
        datasets = [tx, vx]
        datasets_y = [ty, vy]
    else:
        datasets = [tx]
        datasets_y = [ty]

    for k,data in enumerate(datasets):
        vec = []
        i = 1
        # PH image
        for img in data:
            bw_img = (img >= threshold_otsu(img))
            dt_img = distance_transform_edt(bw_img)-distance_transform_edt(~bw_img)
            ph = cripser.computePH(dt_img.astype(np.float64))
            if args.ph=="life":
                v=lifeVect(ph,mx,my,mz,max_life=args.max_life)/args.max_life
            else:
                v=histVect(ph,mx,my,mz,min_life=args.min_life,max_life=args.max_life,
                   n_life_bins=3,n_birth_bins=3,dims=[0,1])
            if args.dim == 2:  ## remove boundary
                v[:,bd,:]=0
                v[:,:,bd]=0
            else:
                v[:,bd,:,:]=0
                v[:,:,bd,:]=0
                v[:,:,:,bd]=0

            vec.append(np.concatenate([np.expand_dims(img, 0),v],axis=0))
            #print(v[v!=0])
            #print(img.shape,v.shape)
            if i%200 == 0:
                print("{} / {}".format(i,len(data)))
            i += 1

        vec = np.array(vec, dtype=np.float32)
        print("output shape:",vec.shape)
        np.savez_compressed(x_filename[k], x=vec, y=datasets_y[k])


