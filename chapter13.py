# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 11:34:34 2020

@author: Sima Soltani
"""

import tensorflow as tf
import numpy as np
np.set_printoptions(precision=3)

a= np.array([1,2,3],dtype=np.int32)
b = [4,5,6]

t_a=tf.convert_to_tensor(a)
t_b=tf.convert_to_tensor(b)

print(t_a)
print(t_b)

t_ones = tf.ones((2,3))
t_ones.shape
t_ones.numpy()

const_tensor = tf.constant([1.2,5,np.pi],dtype = tf.float32)
print(const_tensor)

t_a_new = tf.cast(t_a,tf.int64)
print(t_a_new.dtype)
t= tf.random.uniform(shape =(3,5))
t_tr  = tf.transpose(t)
print(t.shape,' --> ',t_tr.shape)

t = tf.zeros((30,))
t_reshape = tf.reshape (t,shape =(5,6))
print(t_reshape.shape)

t = tf.zeros((1,2,1,4,1))
t_sqz = tf.squeeze(t,axis=(2,4))
print(t.shape,'-->',t_sqz.shape)


tf.random.set_seed(1)
t1 = tf.random.uniform(shape =(5,2),minval =-1.0,maxval =1.0)
t2 = tf.random.normal(shape=(5,2),mean = 0.0, stddev= 1.0)

t3 = tf.multiply(t1,t2).numpy()
print (t3)
t4 = tf.math.reduce_mean(t3, axis=1)
print(t4)
t5 = tf.math.reduce_sum(t3, axis=1)
print(t5)
t6 = tf.linalg.matmul(t1,t2,transpose_b=True)
print(t6)
t7 = tf.linalg.matmul(t1,t2,transpose_a = True)
print(t7.numpy())
norm_t1 = tf.norm(t1,ord=2,axis =1).numpy()
print(norm_t1)

tf.random.set_seed(1)
t = tf.random.uniform((6,))
print(t.numpy())

t_splits = tf.split(t,num_or_size_splits=3)
[item.numpy() for item in t_splits]

tf.random.set_seed(1)
t = tf.random.uniform((5,))
print(t.numpy())
t_splits=tf.split(t,num_or_size_splits=[3,2])
[item.numpy() for item in t_splits]

A = tf.ones((3,))
B=tf.zeros((2,))
c = tf.concat([A,B],axis = 0)
print(c.numpy())

A = tf.ones((3,))
B = tf.zeros((3,))
C = tf.stack([A,B],axis=1)
print(C.numpy())


# Creating a Tensorflow Dataset from existing tensors

a = [1.2, 3.4, 7.5, 4.1, 5.0, 1.0]
ds = tf.data.Dataset.from_tensor_slices(a)
print(ds)
for item in ds:
    print(item)
    
# creat batches 
ds_batch = ds.batch(3)
for i , elem in enumerate(ds_batch,1):
    print('batch {}:'.format(i),elem.numpy())
    
# combining two tensors into a joint dataset
tf.random.set_seed(1)
t_x = tf.random.uniform([4,3],dtype=tf.float32)
t_y = tf.range(4)

ds_x = tf.data.Dataset.from_tensor_slices(t_x)
ds_y = tf.data.Dataset.from_tensor_slices(t_y)
ds_joint = tf.data.Dataset.zip((ds_x,ds_y))
for example in ds_joint:
    print(' x:', example[0].numpy(),
          ' y:',example[1].numpy())
    
ds_joint = tf.data.Dataset.from_tensor_slices((t_x,t_y))
for example in ds_joint:
    print(' x:',example[0].numpy(),
          ' y:', example[1].numpy())

#scale feature to the range of [-1,1)
ds_trans = ds_joint.map(lambda x,y: (x*2-1.0,y))
for example in ds_trans:
    print(' x:', example[0].numpy(),
          ' y:', example[1].numpy())
    
#  shuffle, batch , and repeat
tf.random.set_seed(1)
ds = ds_joint.shuffle(buffer_size = len(t_x))
for example in ds :
    print(' x:',  example[0].numpy(),
          ' y:', example[1].numpy())

ds = ds_joint.batch(batch_size = 3,
                    drop_remainder = False)
batch_x,batch_y = next(iter(ds))
print ('Batch-x:\n',batch_x.numpy())
print('Batch-y:\n',batch_y.numpy())

#repeat
ds = ds_joint.batch(3).repeat(count = 2)
for i ,(batch_x,batch_y) in enumerate(ds):
    print(i,batch_x.shape,batch_y.numpy())
    
ds = ds_joint.repeat(count=2).batch(3)
for i,(batch_x,batch_y) in enumerate(ds):
    print(i,batch_x.shape,batch_y.numpy())
    
## oder 1: Shuffle -> batch-> order
tf.random.set_seed(1)
ds=ds_joint.shuffle(4).batch(2).repeat(3)
for i,(batch_x,batch_y) in enumerate(ds):
    print(i,batch_x.shape,batch_y.numpy())
    
##order 2 : batch-->shuffle--repeat
tf.random.set_seed(1)
ds=ds_joint.batch(2).shuffle(4).repeat(3)
for i,(batch_x,batch_y) in enumerate(ds):
    print(i,batch_x.shape,batch_y.numpy())
    
# create dataset from files on your locla storage disk
import pathlib
imgdir_path = pathlib.Path('data\cat_dog_images')
file_list = sorted([str(path) for path in imgdir_path.glob('*.jpg')])

import matplotlib.pyplot as plt
import os

fig = plt.figure(figsize=(10,5))
for i,file in enumerate(file_list):
    img_raw = tf.io.read_file(file)
    img = tf.image.decode_image(img_raw)
    print('Image shape:', img.shape)
    ax = fig.add_subplot(2,3,i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img)
    ax.set_title(os.path.basename(file),size = 15)
plt.tight_layout()
plt.show()

labels =[1 if 'dog' in os.path.basename(file) else 0 for file in file_list]
print(labels)

ds_files_labels = tf.data.Dataset.from_tensor_slices((file_list,labels))
for item in ds_files_labels:
    print(item[0].numpy(),item[1].numpy())
    
# function of preprocessing an image
def load_and_preprocess(path,label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    image /=255.0
    return image,label

img_height,img_width = 80,120
ds_images_labels = ds_files_labels.map(load_and_preprocess)

fig = plt.figure(figsize = (10,6))
for i,example in enumerate(ds_images_labels):
    ax=fig.add_subplot(2,3,i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(example[0])
    ax.set_title('{}'.format(example[1].numpy()),
                 size = 15)
plt.tight_layout()
plt.show()

#fetching available datasets from the tensorflow_datasets libraray
import tensorflow_datasets as tfds
print(len(tfds.list_builders()))
print(tfds.list_builders())

# fetching dataset
#first approach:
    #1. calling the dataset builder function
    #2. Executing the download_and_prepare() metod
    #3. calling the as_dataset() method
celeba_bldr = tfds.builder('celeb_a')
print(celeba_bldr.info.features)
print(celeba_bldr.info.features['image'])
print(celeba_bldr.info.features['attributes'].keys())
print(celeba_bldr.info.citation)

celeba_bldr.download_and_prepare()
datasets = celeba_bldr.as_dataset(shuffle_files=False)
datasets.keys()

ds_train = datasets['train']
assert isinstance(ds_train,tf.data.Dataset)
example = next(iter(ds_train))
print(type(example))
print(example.keys())

ds_train = ds_train.map(lambda item:(item['image'],
                                     tf.cast(item['attributes']['Male'],
                                             tf.int32)))

ds_train = ds_train.batch(18)
images,labels = next(iter(ds_train))
print(images.shape,labels)
fig = plt.figure(figsize = (12,8))
for  i ,(image,label) in enumerate (zip(images,labels)):
    ax = fig.add_subplot(3,6,i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(image)
    ax.set_title('{}'.format(label),size=15)
plt.show()

mnist,mnist_info = tfds.load('mnist',with_info = True,shuffle_files = False)
print(mnist_info)
print(mnist.keys())
ds_train = mnist['train']
ds_train = ds_train.map(lambda item:(item['image'],item['label']))
ds_train = ds_train.batch(10)
batch = next(iter(ds_train))
print(batch[0].shape,batch[1])
fig = plt.figure(figsize = (15,6))
for i,(image,label) in enumerate(zip(batch[0],batch[1])):
    ax=fig.add_subplot(2,5,i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(image[:,:,0],cmap='gray_r')
    ax.set_title('{}'.format(label),size = 15)
plt.show()


# Building a linear regression model
import numpy as np
X_train = np.arange(10).reshape((10,1))
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0])

plt.plot(X_train,y_train,'o',markersize=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

X_train_norm = (X_train - np.mean(X_train))/np.std(X_train)
ds_train_orig = tf.data.Dataset.from_tensor_slices(
    (tf.cast(X_train_norm,tf.float32),
    tf.cast(y_train,tf.float32)))

#subclass definition for keras model
class MyModel(tf.keras.Model):
    def __init__ (self):
        super(MyModel,self).__init__()
        self.w = tf.Variable(0.0, name = 'weight')
        self.b = tf.Variable(0.0, name = 'bias')
    def call (self,x):
        return self.w * x +self.b
    
model = MyModel()
model.build(input_shape=(None,1))
model.summary()

def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true -y_pred))

def train(model,inputs,outputs,learning_rate):
    with tf.GradientTape() as tape:
        current_loss = loss_fn(model(inputs), outputs)
    dW, db = tape.gradient(current_loss, [model.w, model.b])
    model.w.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)
    
tf.random.set_seed(1)
num_epochs = 200
log_steps = 100
learning_rate = 0.001
batch_size = 1
steps_per_epoch = int(np.ceil(len(y_train)/batch_size))

ds_train = ds_train_orig.shuffle(buffer_size =len(y_train))
ds_train = ds_train.repeat(count = None)
ds_train = ds_train.batch(1)
Ws, bs = [],[]

for i , batch in enumerate(ds_train):
    if i>= steps_per_epoch * num_epochs:
        # break the infinite loop
        break
    Ws.append(model.w.numpy())
    bs.append(model.b.numpy())
    
    bx,by = batch
    loss_val = loss_fn(model(bx),by)

    train(model,bx,by,learning_rate=learning_rate)

    if i%log_steps==0:
        print('Epoch {:4d} Step {:2d} Loss {:6.4f}'.format(
            int(i/steps_per_epoch),i,loss_val))
        
        
print ('Final Parameters: ', model.w.numpy(), model.b.numpy())
X_test = np.linspace(0, 9, num = 100).reshape(-1,1)
X_test_norm = (X_test - np.mean(X_train))/ np.std(X_train)
y_pred = model(tf.cast(X_test_norm, dtype=tf.float32))

fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(1,2,1)
plt.plot(X_train_norm,y_train,'o',markersize=10)
plt.plot(X_test_norm,y_pred,'--',lw=3)
plt.legend(['Training examples','Linear Reg.'],fontsize=15)
ax.set_xlabel('x',size = 15)
ax.set_ylabel('y', size=15)
ax.tick_params(axis='both',which ='major',labelsize = 15)
ax = fig.add_subplot(1,2,2)
plt.plot(Ws, lw=3)
plt.plot(bs, lw=3)
plt.legend(['Weight w', 'Bias unit b'], fontsize = 15)
ax.set_xlabel('Iteration', size=15)
ax.set_ylabel('Value', size = 15)
ax.tick_params(axis='both', which='major',labelsize=15)
plt.show()

# Model training via the .compile() and .fit()
tf.random.set_seed(1)
model = MyModel()
model.compile(optimizer = 'sgd',
              loss = loss_fn,
              metrics=['mae','mse'])
model.fit(X_train_norm,y_train,
          epochs = num_epochs, batch_size = batch_size,
          verbose=1)

print ('Final Parameters: ', model.w.numpy(), model.b.numpy())

X_test = np.linspace(0, 9, num = 100).reshape(-1,1)
X_test_norm = (X_test - np.mean(X_train))/ np.std(X_train)
y_pred = model(tf.cast(X_test_norm, dtype=tf.float32))

fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(1,2,1)
plt.plot(X_train_norm,y_train,'o',markersize=10)
plt.plot(X_test_norm,y_pred,'--',lw=3)
plt.legend(['Training examples','Linear Reg.'],fontsize=15)
ax.set_xlabel('x',size = 15)
ax.set_ylabel('y', size=15)
ax.tick_params(axis='both',which ='major',labelsize = 15)
ax = fig.add_subplot(1,2,2)
plt.plot(Ws, lw=3)
plt.plot(bs, lw=3)
plt.legend(['Weight w', 'Bias unit b'], fontsize = 15)
ax.set_xlabel('Iteration', size=15)
ax.set_ylabel('Value', size = 15)
ax.tick_params(axis='both', which='major',labelsize=15)
plt.show()

# Building a multilayer perceptron for classifying flowers in the Iris dataset
iris,iris_info = tfds.load('iris',with_info=True)
print(iris_info)
tf.random.set_seed(1)
ds_orig=iris['train']
ds_orig = ds_orig.shuffle(150,reshuffle_each_iteration=False)
ds_train_orig=ds_orig.take(100)
ds_test=ds_orig.skip(100)
ds_train_orig = ds_train_orig.map(lambda x:(x['features'],x['label']))
ds_test = ds_test.map(lambda x:(x['features'],x['label']))

#create the NN using Keras
iris_model = tf.keras.Sequential([
    tf.keras.layers.Dense(16,activation='sigmoid',
                          name='fc1', input_shape=(4,)),
    tf.keras.layers.Dense(3, name='fc2',
                          activation = 'softmax')])
iris_model.summary()

#compile this model to specify the loss function, the optimizer, and the metrics for evaluation
iris_model.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
#traing the model
num_epochs=100
training_size = 100
batch_size = 2
steps_per_epoch = np.ceil(training_size/batch_size)

ds_train = ds_train_orig.shuffle(buffer_size=training_size)
ds_train = ds_train.repeat()
ds_train = ds_train.batch(batch_size=batch_size)
ds_train = ds_train.prefetch(buffer_size=1000)

history = iris_model.fit(ds_train, epochs=num_epochs,
                         steps_per_epoch=steps_per_epoch,
                         verbose=0)

hist = history.history

fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(1,2,1)
ax.plot(hist['loss'],lw=3)
ax.set_title('Training loss', size = 15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major',labelsize=15)
ax = fig.add_subplot(1,2,2)
ax.plot(hist['accuracy'],lw=3)
ax.set_title('Training accuracy', size = 15)
ax.set_xlabel('Epoch', size = 15)
ax.tick_params(axis='both',which='major',labelsize=15)
plt.show()

#evaluate on test dataset
results = iris_model.evaluate(ds_test.batch(50),verbose=0)
print('Test loss: {:.4f} Test Acc.: {:.4f}'.format(*results))

# Saving and Reloading the trained model
iris_model.save('iris-classifier.h5',
                overwrite = True,
                include_optimizer = True,
                save_format='h5')

#reload the model
iris_model_new = tf.keras.models.load_model('iris-classifier.h5')
iris_model_new.summary()
results = iris_model_new.evaluate(ds_test.batch(33),verbose=0)
print('Test loss: {:.4f} Test Acc.: {:.4f}'.format(*results))


#logistic function recap
import numpy as np

X = np.array([1,1.4,2.5]) ## first value must be one
w = np.array ([0.4,0.3,0.5])

def net_input (X,w):
    return np.dot (X,w)

def logistic (z):
    return 1.0/(1.0 +np.exp(-z))

def logistic_activation(X,w):
    z = net_input(X,w)
    return logistic(z)

print('P(y=1|x) = %.3f' %logistic_activation(X,w))
    
# w : array with shape=(n_output_units,n_hidden_units+1)
# note that the first column are the bias units
w = np.array([[1.1,1.2,0.8,0.4],
             [0.2,0.4,1.0,0.2],
             [0.6,1.5,1.2,0.7]])

#A : data aray with shape = (n_hidden_units+1,n_samples)
# note that the first column of this array must be 1
A= np.array([[1,.1,.4,.6]])

Z = np.dot(w,A[0])
y_probas = logistic(Z)
print('Net Input :\n', Z)

print(' Output Units:\n', y_probas)
y_class = np.argmax(Z, axis=0)
print('Predictd class label: %d' %y_class)


#code the softmax
def softmax(z):
    return np.exp(z)/np.sum(np.exp(z))

y_probas = softmax(Z)
print('Probabilities:\n', y_probas)
np.sum(y_probas)

import tensorflow as tf
Z_tensor = tf.expand_dims(Z,axis=0) #convert z to tensor and additional dimention resrved for the batch size
tf.keras.activations.softmax(Z_tensor)

##hyperbolic Tangent
# plot the activation functions
import matplotlib .pyplot as plt
def tanh(z):
    e_p= np.exp(z)
    e_m = np.exp(-z)
    return (e_p - e_m)/(e_p +e_m)
z = np.arange(-5,5,0.005)
log_act = logistic(z)
tanh_act = tanh(z)
plt.ylim([-1.5,1.5])
plt.xlabel('net input $z$')
plt.ylabel('activation $\phi(z)$')
plt.axhline(1,color = 'black',linestyle =':')
plt.axhline(0.5, color='black',linestyle = ':')
plt.axhline(0, color='black',linestyle = ':')
plt.axhline(-0.5, color='black',linestyle = ':')
plt.axhline(-1, color='black',linestyle = ':')
plt.plot(z,tanh_act,linewidth=3,linestyle='--',label='tanh')
plt.plot(z,log_act,linewidth=3,label='ligistic')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#using tensorflow activation hyperbolic tan
np.tanh(z)
tf.keras.activations.tanh(z)

from scipy.special import expit #logistic function in scipy
expit(z)

tf.keras.activations.sigmoid(z)


tf.keras.activations.relu(z)
