# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:30:19 2020

@author: Sima Soltani
"""

#compute convolution using numpy
import numpy as np
def conv1d(x,w,p=0,s=1):
    w_rot = np.array(w[::-1])
    x_padded = np.array(x)
    if p>0:
        zero_pad =np.zeros(shape=p)
        x_padded = np.concatenate([zero_pad,x_padded,zero_pad])
    res=[]
    for i in range(0,int(len(x)/s),s):
        res.append(np.sum(x_padded[i:i+w_rot.shape[0]]*w_rot))
    return np.array(res)

##testing
x=[1,3,2,4,5,6,1,3]
w=[1,0,3,1,2]

print('conv1d implementation:',conv1d(x,w,p=2,s=1))

print('convolution result:', np.convolve(x,w,mode='same'))
        

#implement conv2d and compare to the one in scipy.signal
import numpy as np
import scipy.signal

def conv2d (x,w,p=(0,0),s=(1,1)):
    W_rot = np.array(w)[::-1,::-1]
    X_origin = np.array(x)
    n1 = X_origin.shape[0]+2*p[0]
    n2 = X_origin.shape[1]+2*p[1]
    
    X_padded = np.zeros(shape=(n1,n2))
    X_padded[p[0]:p[0]+X_origin.shape[0],
             p[1]:p[1]+X_origin.shape[1]]=X_origin
    
    res = []
    for i in range(0,int((X_padded.shape[0]-W_rot.shape[0])/s[0])+1,s[0]):
        res.append([])
        for j in range(0,int((X_padded.shape[1]-W_rot.shape[1])/s[1]+1),s[1]):
            X_sub = X_padded[i:i+W_rot.shape[0],
                               j:j+W_rot.shape[1]]
            res[-1].append(np.sum(X_sub *W_rot))
    return (np.array(res))


X=[[1,3,2,4],[5,6,1,3],[1,2,0,2],[3,4,3,2]]
W = [[1,0,3],[1,2,1],[0,1,1]]
print('Con2d Implementation :\n', conv2d(X,W,p=(1,1),s=(1,1)))

print('Scipy conv2d results:\n',scipy.signal.convolve2d(X,W,mode='same'))


#Reading an image
import tensorflow as tf

img_raw = tf.io.read_file('data/example-image.png')
img = tf.image.decode_image(img_raw)
print('Image shape :',img.shape)

#reading image into our python session using imageio package
import imageio
img = imageio.imread('data/example-image.png')
print('Image shape:',img.shape)
print('Number of channels:', img.shape[2])
print('Image data type: ',img.dtype)
print(img[100:102,100:102,:])

# try to plot the image
import matplotlib.pyplot as plt

plt.imshow(img)


# using L2 as regularization in NN
from tensorflow import keras

conv_layer = keras.layers.Conv2D(
    filters = 16,
    kernel_size=(3,3),
    kernel_regularizer = keras.regularizers.l2(0.001))

fc_layer = keras.layers.Dense(
    units = 16,
    kernel_regularizer=keras.regularizers.l2(0.001))



#use of cross enthropy loss functions
import tensorflow_datasets as tfds

####### Binary Crossentropy
bce_probabs = tf.keras.losses.BinaryCrossentropy(from_logits = False)
bce_logits = tf.keras.losses.BinaryCrossentropy(from_logits = True)

logits = tf.constant([0.8])
probabs = tf.keras.activations.sigmoid(logits)

tf.print(
    'BCE ( w probabs): {:.4f}'.format(
        bce_probabs(y_true=[1],y_pred=probabs)),
    '(w logits): {:.4f}'.format(
        bce_logits(y_true=[1],y_pred=logits)))

######## Categorical CrossEntropy
cce_probas = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
cce_logits = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

logits = tf.constant([[1.5,0.8,2.1]])
probas = tf.keras.activations.softmax(logits)

tf.print('CCE (w probas):{:.4f}'.format(
    cce_probas(y_true=[0,0,1],y_pred=probas)),
    'CCE (w logits):{:.4f}'.format(
        cce_logits(y_true=[0,0,1],y_pred=logits)))

######## Sparse Categorical CrossEntropy
sp_cce_probas = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
sp_cce_logits = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


tf.print(
    'Sp CCE ( w probabs): {:.4f}'.format(
        sp_cce_probas(y_true=[2],y_pred=probas)),
    '(w logits): {:.4f}'.format(
        sp_cce_logits(y_true=[2],y_pred=logits)))

####### Implementing a deep CNN using Tensorflow

#loading and preprocessing the data
## 3 step loading method:
import tensorflow as tf   
import tensorflow_datasets as tfds
mnist_bldr = tfds.builder('mnist')
mnist_bldr.download_and_prepare()
datasets = mnist_bldr.as_dataset(shuffle_files=False)
mnist_train_orig = datasets['train']
mnist_test_orig = datasets['test']

## split the train/validation datasets 
BUFFER_SIZE = 10000
BATCH_SIZE = 64
NUM_EPOCHS = 20

mnist_train = mnist_train_orig.map(
    lambda item: (tf.cast(item['image'],tf.float32)/255.0,
                  tf.cast(item['label'],tf.int32)))

mnist_test = mnist_test_orig.map(
    lambda item: (tf.cast(item['image'],tf.float32)/255.0,
                  tf.cast(item['label'],tf.int32)))

tf.random.set_seed(1)
mnist_train = mnist_train.shuffle(buffer_size=BUFFER_SIZE,
                                  reshuffle_each_iteration = False)

mnist_valid = mnist_train.take(10000).batch(BATCH_SIZE)
mnist_train = mnist_train.skip(10000).batch(BATCH_SIZE)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(
    filters =32,
    kernel_size=(5,5),
    strides=(1,1), padding = 'same',
    data_format='channels_last',
    name='conv_1', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(
    pool_size=(2,2),name ='pool_1'))
model.add(tf.keras.layers.Conv2D(
    filters=64,
    kernel_size=(5,5),
    strides=(1,1),padding='same',
    name='conv_2',activation='relu'))
model.add(tf.keras.layers.MaxPool2D(
    pool_size=(2,2),name='pool_2'))

model.compute_output_shape(input_shape=(16,28,28,1))

##flatten the output to be able to use it in the Dense layer input
model.add(tf.keras.layers.Flatten())
model.compute_output_shape(input_shape=(16,28,28,1))

##add two dense with a dropout layer in between
model.add(tf.keras.layers.Dense(
    units=1024,name='fc_1',
    activation='relu'))

model.add(tf.keras.layers.Dropout(rate=0.5))

model.add(tf.keras.layers.Dense(
    units=10, name='fc_2',
    activation='softmax'))

model.compute_output_shape(input_shape=(None,28,28,1))

tf.random.set_seed(1)
model.build(input_shape=(None,28,28,1))
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])

history = model.fit(mnist_train,
                    epochs=NUM_EPOCHS,
                    validation_data=mnist_valid,
                    shuffle=True)


#visualize the learning curve
import matplotlib.pyplot as plt
import numpy as np
hist=history.history
x_arr = np.arange(len(hist['loss']))+1
fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(1,2,1)
ax.plot(x_arr,hist['loss'],'-o',label='Train loss')
ax.plot(x_arr,hist['val_loss'],'--', label='Validation loss')
ax.legend(fontsize=15)
ax=fig.add_subplot(1,2,2)
ax.plot(x_arr,hist['accuracy'],'-o',label=' Training Acc.')
ax.plot(x_arr,hist['val_accuracy'],'--',label=' Validation Acc.')
ax.legend(fontsize=15)
plt.show()

test_results = model.evaluate(mnist_test.batch(20))
print('Test Acc.: {:.2f}\%'.format(test_results[1]*100))

batch_test = next(iter(mnist_test.batch(12)))
preds = model(batch_test[0])
tf.print(preds.shape)

preds=tf.argmax(preds,axis=1)
print(preds)

fig = plt.figure(figsize=(12,4))
for i in range(12):
    ax = fig.add_subplot(2,6,i+1)
    ax.set_xticks([]);ax.set_yticks([])
    img = batch_test[0][i,:,:,0]
    ax.imshow(img,cmap='gray_r')
    ax.text(0.9,0.1,'{}'.format(preds[i]),
            size=15, color='blue',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)
plt.show()

# Gender classification from face images using CNN
import tensorflow as tf
import tensorflow_datasets as tfds
celeba_bldr =tfds.builder('celeb_a')
celeba_bldr.download_and_prepare()
celeba = celeba_bldr.as_dataset(shuffle_files=False)

celeba_train = celeba['train']
celeba_valid = celeba['validation']
celeba_test = celeba['test']

def count_items(ds):
    n=0
    for _ in ds:
        n+=1
    return n

print('Train set: {}'.format(count_items(celeba_train)))
print('Validation: {}'.format(count_items(celeba_valid)))
print('Test set: {}'.format(count_items(celeba_test)))

celeba_train = celeba_train.take(16000)
celeba_valid = celeba_valid.take(1000)
print('Train set: {}'.format(count_items(celeba_train)))
print('Validation:{}'.format(count_items(celeba_valid)))


#Augmentation of the data
import matplotlib.pyplot as plt
#take 5 examples
examples=[]
for example in celeba_train.take(5):
    examples.append(example['image'])
    
fig = plt.figure(figsize=(16,8.5))

## column 1: cropping to a bounding-box
ax = fig.add_subplot(2,5,1)
ax.set_title('Crop to \nnbounting-box',size=15)

ax.imshow(examples[0])
ax=fig.add_subplot(2,5,6) 
img_cropped = tf.image.crop_to_bounding_box(
    examples[0],50,20,128,128)
ax.imshow(img_cropped)  
# columns2 : flipping (horizontally) 
ax=fig.add_subplot(2,5,2)
ax.set_title('Flip (horizontal)',size=15)

ax.imshow(examples[1])
ax=fig.add_subplot(2,5,7) 
img_flipped = tf.image.flip_left_right(
    examples[1])
ax.imshow(img_flipped)  

# columns3 : adjust contrast 
ax=fig.add_subplot(2,5,3)
ax.set_title('Adjust contrast',size=15)

ax.imshow(examples[2])
ax=fig.add_subplot(2,5,8) 
img_adj_contrast = tf.image.adjust_contrast(
    examples[2],contrast_factor=2)
ax.imshow(img_adj_contrast) 

# columns4 : adjust brightness
ax=fig.add_subplot(2,5,4)
ax.set_title('adjust brightness',size=15)

ax.imshow(examples[3])
ax=fig.add_subplot(2,5,9) 
img_adj_brightness = tf.image.adjust_brightness(
    examples[3],delta=0.3)
ax.imshow(img_adj_brightness)

# columns5 : cropping from image center 
ax=fig.add_subplot(2,5,5)
ax.set_title('Central crop\nand resize',size=15)

ax.imshow(examples[4])
ax=fig.add_subplot(2,5,10) 
img_center_crop = tf.image.central_crop(
    examples[4],.7)
img_resized = tf.image.resize(
    img_center_crop,size=(218,178))
ax.imshow(img_resized.numpy().astype('uint8'))

plt.show()     

#Random augmentation
tf.random.set_seed(1)
fig = plt.figure(figsize=(14,12))

for i,example in enumerate(celeba_train.take(3)):
    image=example['image']
    
    ax=fig.add_subplot(3,4,i*4+1)
    ax.imshow(image)
    if i == 0:
        ax.set_title('Orig',size=15)
        
    ax=fig.add_subplot(3,4, i*4+2)
    img_crop = tf.image.random_crop(image,size=(178,178,3))
    ax.imshow(img_crop)
    if i ==0:
        ax.set_title('Step1: Random Crop',size=15)
    
    ax=fig.add_subplot(3,4,i*4+3)
    img_flip = tf.image.random_flip_left_right(img_crop)
    ax.imshow(tf.cast(img_flip,tf.uint8))
    if i==0:
        ax.set_title('Step2 : Random flip',size=15)
    
    ax=fig.add_subplot(3,4,i*4+4)
    img_resize = tf.image.resize(img_flip,size=(128,128))
    ax.imshow(tf.cast(img_resize,tf.uint8))
    if i==0:
        ax.set_title('Step3: resize',size=15)
plt.show()

#Define a wriapper function to use the pipeline for 
#data augmentation during model training
def preprocess(example,size=(64,64),mode='train'):
    """
    This function recieves a dictionary containing the keys 'image' and
    'attributes' and return a tuple containing the transformed image 
    and label extracted from dictionary of attributes.

    Parameters
    ----------
    example : type: dictionary of 'image' and 'attribute'
        the dictionary of image and attributes.
    size : type:tuple, shape(dim1,dim2)
        the features of the data example. The default is (64,64).
    mode : string
        

    Returns
    -------
    Tuple containing the transformed image and the label extracted from '
    dictionary of attributes.

    """
    image = example['image']
    label = example['attributes']['Male']
    if mode =='train':
        image_cropped = tf.image.random_crop(
            image,size=(178,178,3))
        image_resize = tf.image.resize(
            image_cropped,size=size)
        image_flip = tf.image.random_flip_left_right(
            image_resize)
        return image_flip/255.,tf.cast(label,tf.int32)
    else: #use center - insted of random crops for non training data
        image_cropped = tf.image.crop_to_bounding_box(
            image,offset_height = 20, offset_width =0,
            target_height = 178, target_width=178)
        image_resize = tf.image.resize(image_cropped,size=size)
    return image_resize/255.,tf.cast(label,tf.int32)       

tf.random.set_seed(1)
ds = celeba_train.shuffle(1000,reshuffle_each_iteration=False)
ds = ds.take(2).repeat(5)
ds = ds.map(lambda x:preprocess(x,size=(178,178),mode = 'train'))

fig = plt.figure(figsize=(15,6))
for j, example in enumerate(ds):
    ax = fig.add_subplot(2,5,j//2+(j%2)*5+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(example[0])
plt.show()

import numpy as np

BATCH_SIZE=32
BUFFER_SIZE = 1000
IMAGE_SIZE=(64,64)
step_per_epoch = np.ceil(16000/BATCH_SIZE)

ds_train = celeba_train.map(
    lambda x:preprocess(x,size=IMAGE_SIZE,mode='train'))

ds_train = ds_train.shuffle(buffer_size=BUFFER_SIZE).repeat()
ds_train = ds_train.batch(BATCH_SIZE)

ds_valid = celeba_valid.map(
    lambda x:preprocess(x,size=IMAGE_SIZE,mode='eval'))
ds_valid=ds_valid.batch(BATCH_SIZE)

#training a CNN gender classifier

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),padding='same',activation='relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Dropout(rate=0.5),
    
    tf.keras.layers.Conv2D(
        64,(3,3),padding='same',activation='relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Dropout(rate=0.5),
    
    
    tf.keras.layers.Conv2D(
        128,(3,3),padding='same',activation='relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    
    
    tf.keras.layers.Conv2D(
        256,(3,3),padding='same',activation='relu')
    ])

model.compute_output_shape(input_shape=(None,64,64,3))

# add a global_average pooling layer
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.compute_output_shape(input_shape=(None,64,64,3))

model.add(tf.keras.layers.Dense(
    units=1,activation=None))

model.compute_output_shape(input_shape=(None,64,64,3))
model.build(input_shape=(None,64,64,3))
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy'])

history = model.fit(ds_train,validation_data=ds_valid,
                    epochs=20,
                    steps_per_epoch=step_per_epoch)


hist= history.history
x_arr = np.arange(len(hist['loss']))+1
fig = plt.figure(figsize=(12,4))
ax=fig.add_subplot(1,2,1)
ax.plot(x_arr,hist['loss'],'-o',label ='Training Loss')
ax.plot(x_arr,hist['val_loss'],'-->',label=' Validation Loss')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch',size=15)
ax.set_ylabel('Loss',size=15)
ax=fig.add_subplot(1,2,2)
ax.plot(x_arr,hist['accuracy'],'-o',label='Training Acc.')
ax.plot(x_arr,hist['val_accuracy'],'-->',label=' Validation Acc.')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch',size= 15)
ax.set_ylabel('Accuracy',size = 15)
plt.show()


# using fit function we can continue training for 10 more epochs
history = model.fit(ds_train,validation_data=ds_valid,
                    epochs=30,initial_epoch=20,
                    steps_per_epoch=step_per_epoch)

hist.update(history.history)
x_arr = np.arange(len(hist['loss']))+1
fig = plt.figure(figsize=(12,4))
ax=fig.add_subplot(1,2,1)
ax.plot(x_arr,hist['loss'],'-o',label ='Training Loss')
ax.plot(x_arr,hist['val_loss'],'-->',label=' Validation Loss')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch',size=15)
ax.set_ylabel('Loss',size=15)
ax=fig.add_subplot(1,2,2)
ax.plot(x_arr,hist['accuracy'],'-o',label='Training Acc.')
ax.plot(x_arr,hist['val_accuracy'],'-->',label=' Validation Acc.')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch',size= 15)
ax.set_ylabel('Accuracy',size = 15)
plt.show()

# Evaluate the model with the test set
ds_test  = celeba_test.map(
    lambda x:preprocess(x,size=IMAGE_SIZE,mode='test')).batch(32)
test_results = model.evaluate(ds_test)
print(' Test accuracy: {:.2f}%'.format(test_results[1]*100))

#visualize 10 test examples
ds = ds_test.unbatch().take(10)

pred_logits = model.predict(ds.batch(10))
probas = tf.sigmoid(pred_logits)
probas = probas.numpy().flatten()*100

fig = plt.figure(figsize = (15,9))
for j,example in enumerate(ds):
    ax = fig.add_subplot(2,5,j+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(example[0])
    if example[1].numpy()==1:
        label='M'
    else:
        label='F'
    ax.text(0.5,-0.15,'GT: {:s}\nPr (Male)={:.0f}%'
            ''.format(label,probas[j]),
            size=16,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)
plt.tight_layout()
plt.show()



#train the model with the whole dataset
celeba_train = celeba['train']
celeba_valid = celeba['validation']
celeba_test = celeba['test']

import numpy as np

BATCH_SIZE=32
BUFFER_SIZE = 1000
IMAGE_SIZE=(64,64)
step_per_epoch = np.ceil(count_items(celeba_train)/BATCH_SIZE)

ds_train = celeba_train.map(
    lambda x:preprocess(x,size=IMAGE_SIZE,mode='train'))

ds_train = ds_train.shuffle(buffer_size=BUFFER_SIZE).repeat()
ds_train = ds_train.batch(BATCH_SIZE)

ds_valid = celeba_valid.map(
    lambda x:preprocess(x,size=IMAGE_SIZE,mode='eval'))
ds_valid=ds_valid.batch(BATCH_SIZE)

#training a CNN gender classifier

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),padding='same',activation='relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Dropout(rate=0.5),
    
    tf.keras.layers.Conv2D(
        64,(3,3),padding='same',activation='relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Dropout(rate=0.5),
    
    
    tf.keras.layers.Conv2D(
        128,(3,3),padding='same',activation='relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    
    
    tf.keras.layers.Conv2D(
        256,(3,3),padding='same',activation='relu')
    ])

model.compute_output_shape(input_shape=(None,64,64,3))

# add a global_average pooling layer
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.compute_output_shape(input_shape=(None,64,64,3))

model.add(tf.keras.layers.Dense(
    units=1,activation=None))

model.compute_output_shape(input_shape=(None,64,64,3))
model.build(input_shape=(None,64,64,3))
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy'])

history = model.fit(ds_train,validation_data=ds_valid,
                    epochs=30,
                    steps_per_epoch=step_per_epoch)


hist= history.history
x_arr = np.arange(len(hist['loss']))+1
fig = plt.figure(figsize=(12,4))
ax=fig.add_subplot(1,2,1)
ax.plot(x_arr,hist['loss'],'-o',label ='Training Loss')
ax.plot(x_arr,hist['val_loss'],'-->',label=' Validation Loss')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch',size=15)
ax.set_ylabel('Loss',size=15)
ax=fig.add_subplot(1,2,2)
ax.plot(x_arr,hist['accuracy'],'-o',label='Training Acc.')
ax.plot(x_arr,hist['val_accuracy'],'-->',label=' Validation Acc.')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch',size= 15)
ax.set_ylabel('Accuracy',size = 15)
plt.show()


# using fit function we can continue training for 10 more epochs
history = model.fit(ds_train,validation_data=ds_valid,
                    epochs=30,initial_epoch=20,
                    steps_per_epoch=step_per_epoch)

hist.update(history.history)
x_arr = np.arange(len(hist['loss']))+1
fig = plt.figure(figsize=(12,4))
ax=fig.add_subplot(1,2,1)
ax.plot(x_arr,hist['loss'],'-o',label ='Training Loss')
ax.plot(x_arr,hist['val_loss'],'-->',label=' Validation Loss')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch',size=15)
ax.set_ylabel('Loss',size=15)
ax=fig.add_subplot(1,2,2)
ax.plot(x_arr,hist['accuracy'],'-o',label='Training Acc.')
ax.plot(x_arr,hist['val_accuracy'],'-->',label=' Validation Acc.')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch',size= 15)
ax.set_ylabel('Accuracy',size = 15)
plt.show()

# Evaluate the model with the test set
ds_test  = celeba_test.map(
    lambda x:preprocess(x,size=IMAGE_SIZE,mode='test')).batch(32)
test_results = model.evaluate(ds_test)
print(' Test accuracy: {:.2f}%'.format(test_results[1]*100))

#visualize 10 test examples
ds = ds_test.unbatch().take(10)

pred_logits = model.predict(ds.batch(10))
probas = tf.sigmoid(pred_logits)
probas = probas.numpy().flatten()*100

fig = plt.figure(figsize = (15,9))
for j,example in enumerate(ds):
    ax = fig.add_subplot(2,5,j+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(example[0])
    if example[1].numpy()==1:
        label='M'
    else:
        label='F'
    ax.text(0.5,-0.15,'GT: {:s}\nPr (Male)={:.0f}%'
            ''.format(label,probas[j]),
            size=16,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)
plt.tight_layout()
plt.show()