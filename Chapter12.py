# -*- codng: utf-8 -*-
"""
Created on Sun Apr 12 09:49:49 2020

@author: Sima Soltani
"""
import os
import struct
import numpy as np

def load_mnist (path, kind='train'):
    """
    Load MNIST data from 'path'"""
    labels_path=os.path.join(path,'%s-labels.idx1-ubyte'%kind)
    images_path=os.path.join(path,'%s-images.idx3-ubyte'%kind)
    
    with open(labels_path,'rb') as lbpath:
        magic, n =struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype = np.uint8)
              
    with open(images_path,'rb') as imgpath:
        magic,num,row,cols=struct.unpack(">IIII",imgpath.read(16))
        images=np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels),784)
        images=((images/255.)-.5)*2
    
    return images,labels


#load train and test data

X_train,y_train = load_mnist('data/',kind='train')
print('Rows: %d,columns: %d'%(X_train.shape[0],X_train.shape[1]))
X_test,y_test = load_mnist('data/',kind='t10k')
print('Rows: %d, Columns: %d'%(X_test.shape[0],X_test.shape[1]))


#visualize the examples
import matplotlib.pyplot as plt
fig,ax =plt.subplots(nrows=2,ncols=5,
                     sharex=True, sharey=True)
ax = ax.flatten()
for i in range (10):
    img =X_train[y_train==i][0].reshape(28,28)
    ax[i].imshow(img,cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

fig,ax=plt.subplots(nrows=5,ncols=5,
                              sharex=True,sharey=True)

ax=ax.flatten()
for i in range(25):
    img=X_train[y_train==7][i].reshape(28,28)
    ax[i].imshow(img,cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])

plt.tight_layout()
plt.show()

#save the data to an archieve file
import numpy as np
np.savez_compressed('mnist_scaled.npz',
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test)

#load the .npz file
mnist = np.load('mnist_scaled.npz')
mnist.files

X_train=mnist['X_train']

X_train,y_train,X_test,y_test = [mnist[f] for f in mnist.files]

#loading dataset using scikit learn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
X,y = fetch_openml('mnist_784',version=1,return_X_y=True)
y = y.astype(int)
X=((X/255.)-0.5)*2
X_train,X_test,y_train,y_test=train_test_split\
    (X,y,test_size=10000,random_state=123,stratify=y)
    
from neuralnet import NeuralNetMLP

nn= NeuralNetMLP(n_hidden=100,
                 l2 = 0.01,
                 epochs = 200,
                 eta = 0.0005,
                 minibatch_size = 100,
                 shuffle = True,
                 seed = 1)

nn.fit(X_train = X_train[:55000],
       y_train = y_train[:55000],
       X_valid = X_train[55000:],
       y_valid = y_train[55000:])

import matplotlib.pyplot as plt
plt.plot(range(nn.epochs),nn.eval_['cost'])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.show()

plt.plot(range(nn.epochs),nn.eval_['train_acc'],label='Training')
plt.plot(range(nn.epochs),nn.eval_['valid_acc'], label='Validation',linestyle ='--')
plt.ylabel('accuracy')
plt.xlabel('Epochs')
plt.legend(loc='best')
plt.show()
y_test_pred = nn.predict(X_test)
acc = (np.sum(y_test == y_test_pred).astype(np.float)/X_test.shape[0])
print(' Test accuracy : %.2f%%'%(acc*100))

miscl_img = X_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]
miscl_lab = y_test_pred[y_test !=y_test_pred][:25]

fig,ax =plt.subplots(nrows = 5,
                     ncols = 5,
                     sharex = True,
                     sharey = True)
ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28,28)
    ax[i].imshow(img,
                 cmap='Greys',
                 interpolation ='nearest')
    ax[i].set_title('%d) t: %d p:%d'
                    %(i+1,correct_lab[i],miscl_lab[i]))
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()