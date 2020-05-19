import logging,os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import *
from keras.utils import to_categorical
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse,os
from datetime import datetime as dt
from datagen3d import DataGenerator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def net3D(in_ch=1,out_ch=10,fc_ch=64,size=16,pool="fc",droprate=0.25):
    model = Sequential()
    model.add(Conv3D(32, kernel_size=3, activation='relu', kernel_initializer='he_uniform', input_shape=(size,size,size,in_ch), padding="same"))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Conv3D(64, kernel_size=3, activation='relu', kernel_initializer='he_uniform', padding="same"))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(MaxPooling3D(pool_size=2))
    model.add(SpatialDropout3D(droprate))
    model.add(Conv3D(64, kernel_size=3, activation='relu', kernel_initializer='he_uniform', padding="same"))
    model.add(keras.layers.normalization.BatchNormalization())
    if pool=="fc":
        model.add(MaxPooling3D(pool_size=2))
        model.add(Dropout(droprate))
        model.add(Flatten())
        model.add(Dense(fc_ch, activation='relu', kernel_initializer='he_uniform'))
    elif pool=="max":
        model.add(GlobalMaxPooling3D())
    else:
        model.add(GlobalAveragePooling3D())

    model.add(Dropout(droprate))
    model.add(Dense(out_ch, activation='softmax'))
    return(model)

def net2D(in_ch=1,out_ch=10,fc_ch=64,size=16,pool="fc",droprate=0.25):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='he_uniform', input_shape=(size,size,in_ch), padding="same"))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Conv2D(64, kernel_size=3, activation='relu', kernel_initializer='he_uniform', padding="same"))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(SpatialDropout2D(droprate))
    model.add(Conv2D(64, kernel_size=3, activation='relu', kernel_initializer='he_uniform', padding="same"))
    model.add(keras.layers.normalization.BatchNormalization())
    if pool=="fc":
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(droprate))
        model.add(Flatten())
        model.add(Dense(fc_ch, activation='relu', kernel_initializer='he_uniform'))
    elif pool=="max":
        model.add(GlobalMaxPooling2D())
        model.add(Dropout(droprate))
    else:
        model.add(GlobalAveragePooling2D())
        model.add(Dropout(droprate))

    model.add(Dense(out_ch, activation='softmax'))
    return(model)

def top2(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')
    parser.add_argument('--train_sample', '-n', default=10000, type=int, help='number of train samples in each class')
    parser.add_argument('--size', '-s', default=16, type=int, help='size of image')
    parser.add_argument('--batchsize', '-b', default=100, type=int)
    parser.add_argument('--epoch', '-e', default=200, type=int)
    parser.add_argument('--gpu', '-g', default=0, type=int)
    parser.add_argument('--conv_dim', '-cd', default=3, type=int, help="dimension: Conv2D or Conv3D")
    parser.add_argument('--fc_ch', '-f', default=64, type=int, help='channel of fc layer')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--droprate', '-dr', type=float, default=0.25)
    parser.add_argument('--used_ch', '-c', type=int, nargs="*", default=None,
                        help='channels used (0: image, 1: H0, 2: H1')
    parser.add_argument('--ph', default="life", choices=["life","hist"])
    parser.add_argument('--pool', default="fc", choices=["fc","avg","max"])
    parser.add_argument('--outdir', '-o', type=str, default=None)
    parser.add_argument('--write_file', '-w', type=str, default="result.csv")
    parser.add_argument('--verbosity', '-v', default=1, type=int)
    parser.add_argument('--random', '-r', action='store_true', help="apply random rotation")
    parser.add_argument('--plot_only', '-p', action='store_true', help="plot and exit")
    args = parser.parse_args()

    PH = True
    dts = dt.now().strftime('%m%d_%H%M')

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[args.gpu], 'GPU')
#    print(physical_devices)

    if args.used_ch is None:
        args.used_ch = [0,1,2,3]

    if args.batchsize > args.train_sample*5:
        args.batchsize = args.train_sample*5

    if args.outdir is None:
        args.outdir = "res_n{}s{}c{}".format(args.train_sample,args.size,"".join([str(i) for i in args.used_ch]))
        if args.random:
            args.outdir += "rnd"

    os.makedirs(args.outdir, exist_ok=True)

    print("Channel {}, Conv dim {}, PH {}".format(args.used_ch,args.conv_dim,PH))

    # dataset loading
    if not PH:
        with h5py.File("full_dataset_vectors.h5", "r") as hf:    
            X_train = hf["X_train"][:]
            Y_train = hf["y_train"][:]
            X_test = hf["X_test"][:] 
            Y_test = hf["y_test"][:]
            X_train = X_train.reshape(-1,16,16,16)
            X_test = X_test.reshape(-1,16,16,16)
    else:
        train_dat = np.load("rmnist3d{}_s{}_train_life.npz".format(args.train_sample,args.size))
        X_train = train_dat['x'].transpose(0,2,3,4,1) # b,x,y,z,c
        Y_train = train_dat['y']
        test_dat = np.load("mnist3d_s{}_test_life.npz".format(args.size))
        X_test = test_dat['x'].transpose(0,2,3,4,1)
        Y_test = test_dat['y']
        X_train = X_train[:,:,:,:,args.used_ch].astype(np.float32)
        X_test = X_test[:,:,:,:,args.used_ch].astype(np.float32)

    if args.conv_dim == 3:
        if(len(X_train.shape)==4):
            X_train = np.expand_dims(X_train,4)
            X_test = np.expand_dims(X_test,4)
        in_ch = X_train.shape[-1]
        model = net3D(in_ch,fc_ch=args.fc_ch,size=args.size,pool=args.pool,droprate=args.droprate)
    else:
        in_ch = X_train.shape[-1]
        model = net2D(in_ch,fc_ch=args.fc_ch,size=args.size,pool=args.pool,droprate=args.droprate)    

    Y_train = to_categorical(Y_train).astype(np.integer)
    yt = Y_test
    Y_test = to_categorical(Y_test).astype(np.integer)
    print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)

    val_ind = range(len(Y_test)//5)
    training_generator = DataGenerator(X_train, Y_train, shuffle=True, 
        random=args.random, batch_size=args.batchsize)
    validation_generator = DataGenerator(X_test[val_ind], Y_test[val_ind], shuffle=False, 
        random=False, batch_size=args.batchsize)

    #plot
    if args.plot_only:
        x,y = training_generator[0]
        for k in range(len(y)):
            print(np.argmax(y[k]),np.mean(x[k,:,:,:,0]),np.max(x[k,:,:,:,0]))
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.voxels(x[k,:,:,:,0]>0.3, edgecolor='k')
            plt.show()
        exit()


    # training
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=args.learning_rate),
                  metrics=['accuracy',top2])

    # history = model.fit(X_train, Y_train,
    #             batch_size=args.batchsize,
    #             epochs=args.epoch,
    #             validation_data=(X_test[val_ind], Y_test[val_ind]),
    #             verbose=args.verbosity)

    history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=args.epoch,
                    verbose=args.verbosity,
#                    use_multiprocessing=True,
                    workers=6)

    score = model.evaluate(X_test, Y_test, verbose=0)
    print(f'Test loss: {score[0]}, Test accuracy: {score[1]}, Top-2: {score[2]}')

    # Plot
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.ylabel('Loss')
    plt.ylim(0.0,1.0)
    plt.xlabel('epoch')
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(args.outdir,"loss3d_{}.jpg".format(dts)))

    # evaluation
    prob = model.predict(X_test, batch_size=args.batchsize)
    pred = np.argsort(-prob, axis=1).astype(np.integer)
    pred1 = K.one_hot(pred[:,0], prob.shape[-1])
#    print(pred[:5],"\n",pred1[:5],"\n",Y_test[:5],"\n",yt[:5])
    correct = [np.sum(Y_test[:,t]*pred1[:,t]) for t in range(10)]
    pred2 = K.one_hot(pred[:,1], prob.shape[-1])
    correct2 = [np.sum(Y_test[:,t]*pred2[:,t]) for t in range(10)]
    total = [np.sum(Y_test[:,t]) for t in range(10)]
    acc = [round(float(correct[t])/total[t],4) for t in range(len(total))]
    acc2 = [round(float(correct[t]+correct2[t])/total[t],4) for t in range(len(total))]
    print('Class accuracy: {}'.format(acc))
    print('Top-2 Class accuracy: {}'.format(acc2))

    if args.write_file:
        w = [str(x) for x in score[1:]+acc+acc2]
        with open(os.path.join(args.outdir,args.write_file),"a") as f:
            f.write(','.join(w))
            f.write('\n')

