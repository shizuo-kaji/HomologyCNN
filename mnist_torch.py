import torch 
import torchvision
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import argparse,os
from datetime import datetime as dt

class Net(nn.Module):
    def __init__(self, in_ch=1, size=28, pool="fc", droprate=0.25):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 3, stride=1, padding=1) 
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1) 
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop2d = nn.Dropout2d(p=droprate)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1) 
        self.bn3 = nn.BatchNorm2d(64)
        self.drop = nn.Dropout(p=droprate)
        self.pool_type = pool
        if pool == "fc":
            self.ldim = (int(size/8)**2) * 64
        else:
            self.ldim = 64
        self.fc1 = nn.Linear(self.ldim, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.drop2d(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.drop2d(x)
        x = F.relu(self.bn3(self.conv3(x)))
        if self.pool_type == "avg":
            x = F.adaptive_avg_pool2d(x, (1, 1)) # global pooling
        elif self.pool_type == "max":
            x = F.adaptive_max_pool2d(x, (1, 1)) # global pooling
        else:
            x = self.pool(x)
        x = self.drop(x)
        x = x.view(-1, self.ldim)
        x = self.fc1(x)
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')
    parser.add_argument('--train_sample', '-n', default=1, type=int, help='number of train samples in each class')
    parser.add_argument('--batchsize', '-b', default=3, type=int)
    parser.add_argument('--gpu', '-g', default=0, type=int)
    parser.add_argument('--epoch', '-e', default=1000, type=int)
    parser.add_argument('--size', '-s', default=28, type=int, help='size of image')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--droprate', '-dr', type=float, default=0.25)
    parser.add_argument('--used_ch', '-c', type=int, nargs="*", default=None,
                        help='channels used (0: image, 1: H0, 2: H1')
    parser.add_argument('--ph', default="life", choices=["life","hist"])
    parser.add_argument('--pool', default="fc", choices=["fc","avg","max"])
    parser.add_argument('--plot_only', '-p', action='store_true')
    parser.add_argument('--outdir', '-o', type=str, default="result")
    parser.add_argument('--write_file', '-w', type=str, default="")
    args = parser.parse_args()

    size_str = "x{}".format(args.size)
    dts = dt.now().strftime('%m%d_%H%M')
    os.makedirs(args.outdir, exist_ok=True)

    if args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')

    filename = ["rmnist{}_s{}_train_{}.npz".format(args.train_sample,args.size,args.ph),"mnist_s{}_test_{}.npz".format(args.size,args.ph)]
    train_dat = np.load(filename[0])
    test_dat =np.load(filename[1])
    if args.used_ch is None:
        args.used_ch = range(train_dat['x'].shape[1])

    print("n: {}, epoch: {}, used_ch: {}, {}: {}".format(args.train_sample,args.epoch,args.used_ch,filename[0],train_dat['x'].shape))

    if args.plot_only:
        pdat = train_dat['x']
        k=10
        fig = plt.figure(figsize=(10, 7))
        grid = ImageGrid(fig, 111,nrows_ncols=(3, k),axes_pad=0.1)
        for ax, im in zip(grid, [pdat[i,j] for j in range(3) for i in range(k)]):
            ax.imshow(im)
        print(np.max(pdat[:k,2],axis=(1,2)))
        plt.show()
        exit()

    train_x = torch.from_numpy(train_dat['x'][:,args.used_ch,:,:].astype(np.float32))
    train_y = torch.from_numpy(train_dat['y'].astype(np.int64))
    test_x = torch.from_numpy(test_dat['x'][:,args.used_ch,:,:].astype(np.float32))
    test_y = torch.from_numpy(test_dat['y'].astype(np.int64))
#    print(test_x.shape,test_y.shape)
    in_ch = train_x.shape[1]

    train = torch.utils.data.TensorDataset(train_x,train_y)
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batchsize, shuffle=True)
    test = torch.utils.data.TensorDataset(test_x, test_y)
    test_loader = torch.utils.data.DataLoader(test, batch_size=int(100*(28*28)/(args.size**2)), shuffle=False)

    net = Net(in_ch,size=args.size,pool=args.pool,droprate=args.droprate)
    net = net.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    # training loop
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    for i in range(args.epoch):
        running_loss = 0.0
        total,correct = 0,0
        for (x,y) in train_loader:
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            out = net(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        
        # validation
        if ((i+1) % int(args.epoch/20)) == 0:
            train_loss.append(100*running_loss/len(train_loader))
            train_acc.append(correct/total)
            #
            total,correct = 0,0
            with torch.no_grad():
                for (x,y) in test_loader:
                    net.eval()       
                    x = x.to(device)
                    y = y.to(device)
                    out = net(x)
                    loss_test = criterion(out, y)
                    _, predicted = torch.max(out.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
            val_loss.append(100*loss_test.item()/len(test_loader))
            val_acc.append(correct/total)
            print("epoch {}, train loss {:.4f}, train acc {:.3f}, test loss {:.4f}, test acc {:.3f}".format(i+1,train_loss[-1],train_acc[-1],val_loss[-1],val_acc[-1]))
            net.train()

    plt.plot(val_loss,label = "val loss")
    plt.plot(val_acc,label = "val acc")
    plt.plot(train_loss,label = "train loss")
    plt.plot(train_acc,label = "train acc")
    plt.ylim(0.0,1.1)
    plt.xlabel('epoch')
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(args.outdir,"loss_{}.jpg".format(dts)))

    # evaluate on test dataset
    correct = [0]*10
    correct2 = [0]*10 # second prediction
    total = [0]*10

    with torch.no_grad():
        net.eval()
        for (x, y) in test_loader:
            x = x.to(device)
            outputs = net(x)
            _, predicted = torch.topk(outputs.data, 2, dim=1)
            for t,y in zip(y,predicted):
                total[t] += 1
                correct[t] += (t==y[0])
                correct2[t] += (t==y[1])


    acc = round( (100.0 * sum(correct)/sum(total)).item(), 4)
    acc2 = round( (100.0 * (sum(correct)+sum(correct2))/sum(total)).item(), 4)
    acc_each = [round(float(correct[i])/total[i],4) for i in range(len(total))]
    acc2_each = [round(float(correct[i]+correct2[i])/total[i],4) for i in range(len(total))]
    print('Num: {}, Accuracy: {:.2f} %%, Top-2 Accuracy {:.2f} %%'.format(sum(total), acc, acc2))
    print('Class Accuracy: {}'.format(acc_each))
    print('Top-2 Class Accuracy: {}'.format(acc2_each))
    if args.write_file:
        w = [str(x) for x in [acc,acc2]+acc_each+acc2_each]
        with open(os.path.join(args.outdir,args.write_file),"a") as f:
            f.write(','.join(w))
            f.write('\n')
