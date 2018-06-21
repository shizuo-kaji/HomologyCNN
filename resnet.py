import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

def add_noise(h, sigma):  
    xp = chainer.cuda.get_array_module(h)
    if chainer.config.train:
        return h + sigma * xp.random.randn(*h.shape)
    else:
        return h

class BottleNeck(chainer.Chain):
    def __init__(self, n_in, n_mid, n_out, stride=1, use_conv=False, activation=F.relu, dtype=np.float32):
        w = chainer.initializers.HeNormal(dtype=dtype)
        bias = chainer.initializers.Zero(dtype)
        self.activation = activation
        super(BottleNeck, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(n_in, n_mid, 1, stride, 0, True, w, initial_bias=bias)
            self.bn1 = L.BatchNormalization(n_mid, dtype=dtype)
            self.conv2 = L.Convolution2D(n_mid, n_mid, 3, 1, 1, True, w, initial_bias=bias)
            self.bn2 = L.BatchNormalization(n_mid, dtype=dtype)
            self.conv3 = L.Convolution2D(n_mid, n_out, 1, 1, 0, True, w, initial_bias=bias)
            self.bn3 = L.BatchNormalization(n_out, dtype=dtype)
            if use_conv:
                self.conv4 = L.Convolution2D(n_in, n_out, 1, stride, 0, True, w, initial_bias=bias)
                self.bn4 = L.BatchNormalization(n_out, dtype=dtype)
        self.use_conv = use_conv

    def __call__(self, x):
        h = self.activation(self.bn1(self.conv1(x)))
        h = self.activation(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        return h + self.bn4(self.conv4(x)) if self.use_conv else h + x


class Block(chainer.ChainList):
    def __init__(self, n_in, n_mid, n_out, n_bottlenecks, stride=2, activation=F.relu, dtype=np.float32):
        super(Block, self).__init__()
        self.add_link(BottleNeck(n_in, n_mid, n_out, stride, True, activation, dtype))
        for _ in range(n_bottlenecks - 1):
            self.add_link(BottleNeck(n_out, n_mid, n_out, 1 ,False, activation, dtype))

    def __call__(self, x):
        for f in self:
            x = f(x)
        return x


class ResNet(chainer.Chain):
    def __init__(self, num_class, n_blocks=[3, 4, 6, 3],**kwargs):
        super(ResNet, self).__init__()
        self.num_class = num_class
        params = kwargs.pop('params')
        self.activation = params['activation']
        self.dtype = params['dtype']
        self.noise = params['noise']
        self.num_class = num_class
        self.drop = lambda x: F.dropout(x, ratio=params['drop_ratio'])
        first_ksize = params['first_ksize']
        w = chainer.initializers.HeNormal(dtype=self.dtype)
        bias = chainer.initializers.Zero(dtype=self.dtype)
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, first_ksize, 1, 0, True, w, initial_bias=bias)
            self.bn2 = L.BatchNormalization(64, dtype=self.dtype)
            self.res3 = Block(64, 64, 256, n_blocks[0], 1, self.activation, self.dtype)
            self.res4 = Block(256, 128, 512, n_blocks[1], 2, self.activation, self.dtype)
            self.res5 = Block(512, 256, 1024, n_blocks[2], 2, self.activation, self.dtype)
            self.res6 = Block(1024, 512, 2048, n_blocks[3], 2, self.activation, self.dtype)
            self.fc7 = L.Linear(None, num_class, initialW=w, initial_bias=bias)

    def __call__(self, x):
        h = x
        if self.noise>0:
            h = add_noise(h,self.noise)
        h = F.cast(h,self.dtype)
        h = self.drop(self.bn2(self.conv1(h)))
        h = self.activation(h)            
        h = self.drop(self.res3(h))
        h = self.drop(self.res4(h))
        h = self.drop(self.res5(h))
        h = self.drop(self.res6(h))
        h = self.drop(F.average_pooling_2d(h, h.shape[2:]))
        h = self.fc7(h)
        return h
