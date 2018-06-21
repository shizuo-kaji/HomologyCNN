import chainer
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L
import numpy as np

def add_noise(h, sigma):
    xp = chainer.cuda.get_array_module(h)
    if chainer.config.train:
        return h + sigma * xp.random.randn(*h.shape)
    else:
        return h

def reshape(x, channels):
  if x.shape[1] < channels:
    xp = chainer.cuda.get_array_module(x)
    p = xp.zeros((x.shape[0], channels - x.shape[1], x.shape[2], x.shape[3]), dtype=x.dtype)
    x = chainer.functions.concat((x, p), axis=1)
  return x


## TODO: dropout?
class ResBlock(chainer.Chain):
    def __init__(self, ch, activation=F.relu, dtype=np.float32):
        super(ResBlock, self).__init__()
        self.activation = activation
        self.dtype = dtype
        w = I.HeNormal(dtype=self.dtype)
        bias = chainer.initializers.Zero(self.dtype)
        with self.init_scope():
            self.c0 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w, initial_bias=bias)
            self.c1 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w, initial_bias=bias)
            self.norm0 = L.BatchNormalization(ch, dtype=dtype)
            self.norm1 = L.BatchNormalization(ch, dtype=dtype)

    def __call__(self, x):
        h = self.c0(x)
        h = self.norm0(h)
        h = self.activation(h)
#        h = self.drop(h)
        h = self.c1(h)
        h = self.norm1(h)
        return h + x

class NIN(chainer.Chain):

    """Network-in-Network model."""

    def __init__(self, num_class, **kwargs):
        super(NIN, self).__init__()
        params = kwargs.pop('params')
        self.activ = params['activation']
        self.chs = params['channels']
        self.dtype = params['dtype']
        self.noise = params['noise']
        self.num_class = num_class
        self.drop = lambda x: F.dropout(x, ratio=params['drop_ratio'])
        self.n_resblock = params['n_resblock']
        self.stride = params['stride']        
        first_ksize = params['first_ksize']
        ksize = params['ksize']
        conv_init = I.HeNormal(dtype=self.dtype)  # MSRA scaling
        bias = chainer.initializers.Zero(self.dtype)

        with self.init_scope():
            setattr(self, 'mlpconv' + str(0), L.MLPConvolution2D(
                None, (self.chs[0], self.chs[0], self.chs[0]), first_ksize, stride=self.stride, pad=first_ksize//2, conv_init=conv_init, bias_init=bias))
            self.norm0 = L.BatchNormalization(self.chs[0], dtype=self.dtype)
            for i in range(1,len(self.chs)):
                setattr(self, 'mlpconv' + str(i), L.MLPConvolution2D(
                    None, (self.chs[i], self.chs[i], self.chs[i]), ksize, stride=self.stride,pad=ksize//2, conv_init=conv_init, bias_init=bias))
                setattr(self, 'norm' + str(i), L.BatchNormalization(self.chs[i], dtype=self.dtype))
            for i in range(self.n_resblock):
                setattr(self, 'res' + str(i), ResBlock(self.chs[-1], self.activ, dtype=self.dtype))
            self.last = L.Convolution2D(self.chs[-1], num_class, ksize, pad=ksize//2, initialW=conv_init, initial_bias=bias)

    def __call__(self, x):
        h = x
        if self.noise>0:
            h = add_noise(h,self.noise)
        h = F.cast(h,self.dtype)
        for i in range(len(self.chs)):
            h = getattr(self, 'mlpconv' + str(i))(h)
            if self.stride==1:
                h = F.max_pooling_2d(h, 3, stride=2)         ## when stride=1 for MLP
            h = getattr(self, 'norm' + str(i))(h)
            h = self.drop(h) ## where to put dropout?  BN->drop or drop->BN?
            h = self.activ(h)
        for i in range(self.n_resblock):
            h = self.activ(getattr(self, 'res' + str(i))(h))            
        h = self.last(h)
        h = F.reshape(F.average_pooling_2d(h, h.shape[2]), (len(x), self.num_class))
        return h

class NINres(chainer.Chain):

    """residual version of Network-in-Network model."""

    def __init__(self, num_class, **kwargs):
        super(NINres, self).__init__()
        params = kwargs.pop('params')
        self.activ = params['activation']
        self.chs = params['channels']
        self.dtype = params['dtype']
        self.noise = params['noise']
        self.stride = params['stride']        
        self.num_class = num_class
        self.n_resblock = params['n_resblock']
        self.drop = lambda x: F.dropout(x, ratio=params['drop_ratio'])
        first_ksize = params['first_ksize']
        ksize = params['ksize']
        conv_init = I.HeNormal(dtype=self.dtype)  # MSRA scaling
        bias = chainer.initializers.Zero(self.dtype)

        with self.init_scope():
            setattr(self, 'mlpconv' + str(0), L.MLPConvolution2D(
                None, (self.chs[0], self.chs[0], self.chs[0]), first_ksize, stride=self.stride, pad=first_ksize//2, conv_init=conv_init, bias_init=bias))
            self.norm0 = L.BatchNormalization(self.chs[0], dtype=self.dtype)
            for i in range(1,len(self.chs)):
                setattr(self, 'mlpconv' + str(i), L.MLPConvolution2D(
                    None, (self.chs[i], self.chs[i], self.chs[i]), ksize, pad=ksize//2, conv_init=conv_init, bias_init=bias))
                setattr(self, 'norm' + str(i), L.BatchNormalization(self.chs[i], dtype=self.dtype))
            for i in range(self.n_resblock):
                setattr(self, 'res' + str(i), ResBlock(self.chs[-1], self.activ, dtype=self.dtype))
            self.last = L.Convolution2D(self.chs[-1], num_class, ksize, pad=ksize//2, initialW=conv_init, initial_bias=bias)

    def __call__(self, x):
        h = x
        if self.noise>0:
            h = add_noise(h,self.noise)
        h = F.cast(h,self.dtype)
        h = self.mlpconv0(h)
        if self.stride==1:
            h = F.max_pooling_2d(h, 3, stride=2)         ## when stride=1 for MLP
        h = self.activ(self.norm0(h))
        ## Conv+Drop+Norm+Add+Pool+Active      drop must be before Add!
        for i in range(1,len(self.chs)):
            y = h
            h = getattr(self, 'mlpconv' + str(i))(h)
            h = getattr(self, 'norm' + str(i))(h)
            h = self.drop(h)   ## some say dropout should be after BN, but...
            h += reshape(y, h.shape[1])
            h = F.max_pooling_2d(h, 3, stride=2)
            h = self.activ(h)
        for i in range(self.n_resblock):
            h = self.activ(getattr(self, 'res' + str(i))(h))            
        h = self.last(h)
        h = F.reshape(F.average_pooling_2d(h, h.shape[2]), (len(x), self.num_class))
        return h
        