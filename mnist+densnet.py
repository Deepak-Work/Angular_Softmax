
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import numpy
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from sklearn.manifold import TSNE
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
# My model
from LargeMargin import DenseNet as dn
import torch.optim as optim
from numpy import linalg
import argparse

# Hyper Parameters
EPOCH = 5
# train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 128
LR = 0.001              # learning rate
DOWNLOAD_MNIST = True   # set to False if you have downloaded

#Normalize with the grayscale channel's mean and var


parser = argparse.ArgumentParser()
parser.add_argument('--margin',default=2,type=int)
args = parser.parse_args()
trainTransform = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
    
testTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])



# Mnist digits dataset
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=trainTransform,    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,                        # download it if you don't have it


)


test_data = torchvision.datasets.MNIST(
    root='./mnist/', 
    train=False,
    transform = testTransform)

# plot one example
print(train_data.train_data.size())                 # (60000, 28, 28)
print(train_data.train_labels.size())               # (60000)

# convert test data into Variable, pick 2000 samples to speed up testing
# test_data = torchvision.datasets.MNIST(root='./mnist/', train=False,transform=testTransform)

train_loader = DataLoader(
    dataset=train_data, 
    batch_size=BATCH_SIZE, 
    shuffle=True)

testLoader = DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE, 
        shuffle=False)


model = dn(num_init_features=10,margin=args.margin)
model.cuda()

# Hyper Parameters
EPOCH = 5

BATCH_SIZE = 128
LR = 0.0001

optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted


def accuracy():
    model.eval()
    test_loss = 0
    correct = 0
    i=0
    ret = np.zeros((10000,10))
    label = np.zeros(10000)
    for data, target in testLoader:
        data, target = Variable(data, volatile=True), Variable(target)
        data = data.cuda()
        target = target.cuda()
        output = model(x=data)
        test_loss += loss_func(output,target= target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
        ret[i:i+128,:] =output.cpu().detach().numpy() 
        label[i:i+128] =pred 
        i+=128

    test_loss = test_loss
    test_loss /= len(testLoader) # loss function already averages over batch size
    print("Accuracy")
    print(correct)
    print("Loss")
    print(test_loss)
    return ret,label



# training and testing
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):   # divide batch data, normalize x when iterate train_loader
        b_x = Variable(x)   # batch x
        b_y = Variable(y)   # batch y
        #print(np.shape(b_x))
        #print(np.shape(b_y))
        output = model(x = b_x.cuda(),target =b_y.cuda())               # cnn output

        loss = loss_func(output.cuda(),b_y.cuda())
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

       
        if step % BATCH_SIZE == 0 :
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, step * len(b_x), len(train_loader.dataset),
            100. * step / len(train_loader), loss.data[0]))
    ret,label = accuracy()
    print(ret.shape)
ret=np.asarray(ret)
label=np.asarray(label)
np.savetxt('embed.tsv',ret,delimiter='\t')
np.savetxt('metadata.tsv',label,delimiter='\t')
embed = TSNE(n_components=2).fit_transform(ret)
plt.scatter(embed[:,0],embed[:,1],c = label,s = 1, marker='x')
plt.show()
model.eval()




def plot_kernels(tensor, num_cols=3):

    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[i][0],cmap='gray')
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

mm= model.double()
filters = mm.modules
body_model = [i for i in mm.children()]
body_model = body_model[0][3][3][4]
layer1 = body_model
tensor = layer1.weight.data.cpu().numpy()
plot_kernels(tensor)

