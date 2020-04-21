# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 11:41:23 2020

@author: Sima Soltani
"""
# TF version 1.x style
import tensorflow as tf
g = tf.Graph()
with g.as_default():
    a = tf.constant(1,name='a')
    b = tf.constant(2,name='b')
    c = tf.constant(3,name='c')
    z = 2*(a-b)+c
    
with tf.compat.v1.Session(graph=g) as sess:
    print('Result:z=',sess.run(z))
    
#Migrating a graph to Tensorflow 2

##TF version 2.x style
a = tf.constant(1,name ='a')
b = tf.constant(2,name='b')
c = tf.constant(3,name='c')
z=2*(a-b)+c
tf.print('Result: z=',z)

# Loading input data into model
##TF v1.x style
g = tf.Graph()
with g.as_default():
    a = tf.compat.v1.placeholder(shape = None,
                                 dtype = tf.int32,
                                 name = 'tf_a')
    b = tf.compat.v1.placeholder(shape=None,
                                 dtype=tf.int32,
                                 name = 'tf_b')
    c= tf.compat.v1.placeholder(shape=None,
                                dtype=tf.int32,
                                name='tf_c')
    z=2*(a-b)+c
    
with tf.compat.v1.Session(graph=g) as sess:
    feed_dict = {a:1,b:2,c:3}
    print('Result:z =',sess.run(z,feed_dict=feed_dict))
    
##TF v2 style
def compute_z(a,b,c):
    r1=tf.subtract(a,b)
    r2 = tf.multiply(2,r1)
    z = tf.add(r2,c)
    return z

tf.print('Scalar Inputs:', compute_z(1,2,3))
tf.print('Rank 1 Inputes:',compute_z([1],[2],[3]))    
tf.print('Rnak 2 Inputs:', compute_z([[1]],[[2]],[[3]]))


@tf.function
def compute_z(a,b,c):
    r1=tf.subtract(a,b)
    r2 = tf.multiply(2,r1)
    z=tf.add(r1,r2)
    return(z)

tf.print('Scalar Inputs:', compute_z(1,2,3))
@tf.function(input_signature=(tf.TensorSpec(shape=[None],
                                            dtype=tf.int32),
                              tf.TensorSpec(shape=[None],
                                            dtype=tf.int32),
                              tf.TensorSpec(shape=[None],
                                            dtype=tf.int32),))
def compute_z(a,b,c):
    r1=tf.subtract(a,b)
    r2=tf.multiply(2,r1)
    z=tf.add(r2,c)
    return z

tf.print('Rank 1 Input:', compute_z([1],[2],[3]))
tf.print('Rank 1 Inputs:',compute_z([1,2],[2,4],[3,6]))

# Tensorflow Variable objects for storing and updating model parameters
a = tf.Variable(initial_value = 3.14, name ='var_a')
print(a)
b = tf.Variable(initial_value=[1,2,3], name ='var_b')
print(b)
c = tf.Variable(initial_value=[True,False],dtype=tf.bool)
print(c)
d=tf.Variable(initial_value=['abc'],dtype=tf.string)
print(d)


#Define a non-trainable variable
w=tf.Variable([1,2,3],trainable=False)
print(w.trainable)

print(w.assign([3,1,4],read_value=True))
w.assign_add([2,-1,2],read_value=False)
print(w.value())


#Create a Variable with Glorot initialization
tf.random.set_seed(1)
init=tf.keras.initializers.GlorotNormal()
tf.print(init(shape=(3,)))

#initialize a cvariable of shape 2x3
v=tf.Variable(init(shape=(2,3)))
tf.print(v)


class MyModule(tf.Module):
    def __init__(self):
        init=tf.keras.initializers.GlorotNormal()
        self.w1 = tf.Variable(init(shape=(2,3)),
                              trainable=True)
        self.w2 = tf.Variable(init(shape=(1,2)),
                              trainable = False)


m=MyModule()
print('All module variables:',[v.shape for v in m.variables])
print('Trainable variable:',[v.shape for v in m.trainable_variables])


@tf.function
def f(x):
    w=tf.Variable([1,2,3])

f([1])

#define variable outside the decorated function and use it inside
w = tf.Variable(tf.random.uniform((3,3)))
@tf.function
def compute_z(x):
    return tf.matmul(w,x)

x=tf.constant([[1],[2],[3]],dtype=tf.float32)
tf.print(compute_z(x))


# compute gradients via automatic differentiation and GradientTape
w= tf.Variable(1.0)
b= tf.Variable(0.5)
print(w.trainable,b.trainable)
x = tf.convert_to_tensor([1.4])
y = tf.convert_to_tensor([2.1])
with tf.GradientTape() as tape:
    z=tf.add(tf.multiply(w,x),b)
    loss = tf.reduce_sum(tf.square(y-z))
dloss_dw = tape.gradient(loss,w)
tf.print('dL/dw:',dloss_dw)

#verify the computed gradient
tf.print(2*x*(w*x+b-y))

#computing gradients with respect to non-trainable tensors
with tf.GradientTape() as tape:
    tape.watch(x)
    z = tf.add(tf.multiply(w,x),b)
    loss = tf.reduce_sum(tf.square(y-z))
dloss_dx = tape.gradient(loss,x)
tf.print('dL/dx:',dloss_dx)

#keeping resources for multiple gradient computations
with tf.GradientTape(persistent=True) as tape:
    z = tf.add(tf.multiply(w,x),b)
    loss= tf.reduce_sum(tf.square(y-z))
dloss_dw = tape.gradient(loss,w)
tf.print('dL/dw:',dloss_dw)
dloss_db = tape.gradient(loss,b)
tf.print('dL/db:',dloss_db)

#define an optimizer to apply the model parameters
optimizer = tf.keras.optimizers.SGD()
optimizer.apply_gradients(zip([dloss_dw,dloss_db],[w,b]))
tf.print('updated w:',w)
tf.print('Updated b:',b)

#Simplifying implementations of common architectures via Keras API

#sample of a NN with two densly connected layers
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=16,activation ='relu'))
model.add(tf.keras.layers.Dense(units=32, activation ='relu'))
##late bvariable creation
model.build(input_shape=(None,4))
model.summary()
## printing variables of model
for v in model.variables:
    print('{:20s}'.format(v.name),v.trainable,v.shape)
    
# configure the layers by applying activation functions, variable initilizers
# or regularization methods
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(
    units = 16,
    activation = tf.keras.activations.relu,
    kernel_initializer = tf.keras.initializers.glorot_uniform(),
    bias_initializer = tf.keras.initializers.constant(2.0)
    ))

model.add(tf.keras.layers.Dense(
    units = 32,
    activation = tf.keras.activations.sigmoid,
    kernel_regularizer = tf.keras.regularizers.l1
    ))

model.compile(
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics = [tf.keras.metrics.Accuracy(),
               tf.keras.metrics.Precision(),
               tf.keras.metrics.Recall(),])

# create a example dataset for XOR problem
import numpy as np
import matplotlib.pyplot as plt

tf.random.set_seed(1)
np.random.seed(1)

x=np.random.uniform(low=-1,high=1,size=(200,2))
y = np.ones(len(x))
y[x[:,0]*x[:,1]<0]=0

x_train = x[:100,:]
y_train = y[:100]
x_valid = x[100:,:]
y_valid = y[100:]


fig = plt.figure(figsize = (6,6))
plt.plot(x[y==0,0],x[y==0,1],
         'o',alpha = .75,markersize = 10)
plt.plot(x[y==1,0],x[y==1,1],
         '<',alpha=0.75,markersize = 10)
plt.xlabel(r'$x_1$',size =15)
plt.ylabel(r'$x_2$',size = 15)
plt.show()

# a simple model as a base line
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(
    units=1,
    input_shape = (2,),
    activation = 'sigmoid'))
model.summary()

model.compile (optimizer = tf.keras.optimizers.SGD(),
               loss = tf.keras.losses.BinaryCrossentropy(),
               metrics=[tf.keras.metrics.BinaryAccuracy()])
hist = model.fit(x_train,y_train,
                 validation_data = (x_valid,y_valid),
                 epochs=200, batch_size = 2, verbose=0)

from mlxtend.plotting import plot_decision_regions

history = hist.history
fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(1,3,1)
plt.plot(history['loss'],lw=4)
plt.plot(history['val_loss'],lw=4)
plt.legend(['Train loss','Validation loss'],fontsize=15)
ax.set_xlabel('Epochs',size=15)

ax=fig.add_subplot(1,3,2)
plt.plot(history['binary_accuracy'],lw=4)
plt.plot(history['val_binary_accuracy'],lw=4)
plt.legend(['Train acc.','Validation Acc.'],fontsize=15)
ax.set_xlabel('Epochs',size=15)

ax=fig.add_subplot(1,3,3)
plot_decision_regions(X=x_valid,y=y_valid.astype(np.integer),
                      clf=model)
ax.set_xlabel(r'$x_1$',size=15)
ax.xaxis.set_label_coords(1,-0.025)
ax.set_ylabel(r'$x_2$',size=15)
ax.yaxis.set_label_coords(-0.025,1)
plt.show()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(
    units=4,
    input_shape = (2,),
    activation = 'relu'))

model.add(tf.keras.layers.Dense(
    units=1,
    activation = 'sigmoid'))
model.summary()

model.compile (optimizer = tf.keras.optimizers.SGD(),
               loss = tf.keras.losses.BinaryCrossentropy(),
               metrics=[tf.keras.metrics.BinaryAccuracy()])
hist = model.fit(x_train,y_train,
                 validation_data = (x_valid,y_valid),
                 epochs=200, batch_size = 2, verbose=0)

history = hist.history
fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(1,3,1)
plt.plot(history['loss'],lw=4)
plt.plot(history['val_loss'],lw=4)
plt.legend(['Train loss','Validation loss'],fontsize=15)
ax.set_xlabel('Epochs',size=15)

ax=fig.add_subplot(1,3,2)
plt.plot(history['binary_accuracy'],lw=4)
plt.plot(history['val_binary_accuracy'],lw=4)
plt.legend(['Train acc.','Validation Acc.'],fontsize=15)
ax.set_xlabel('Epochs',size=15)

ax=fig.add_subplot(1,3,3)
plot_decision_regions(X=x_valid,y=y_valid.astype(np.integer),
                      clf=model)
ax.set_xlabel(r'$x_1$',size=15)
ax.xaxis.set_label_coords(1,-0.025)
ax.set_ylabel(r'$x_2$',size=15)
ax.yaxis.set_label_coords(-0.025,1)
plt.show()


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(
    units=4,
    input_shape = (2,),
    activation = 'relu'))
model.add(tf.keras.layers.Dense(
    units=4,
    activation = 'relu'))
model.add(tf.keras.layers.Dense(
    units=1,
    activation = 'sigmoid'))
model.summary()

model.compile (optimizer = tf.keras.optimizers.SGD(),
               loss = tf.keras.losses.BinaryCrossentropy(),
               metrics=[tf.keras.metrics.BinaryAccuracy()])
hist = model.fit(x_train,y_train,
                 validation_data = (x_valid,y_valid),
                 epochs=200, batch_size = 2, verbose=0)

history = hist.history
fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(1,3,1)
plt.plot(history['loss'],lw=4)
plt.plot(history['val_loss'],lw=4)
plt.legend(['Train loss','Validation loss'],fontsize=15)
ax.set_xlabel('Epochs',size=15)

ax=fig.add_subplot(1,3,2)
plt.plot(history['binary_accuracy'],lw=4)
plt.plot(history['val_binary_accuracy'],lw=4)
plt.legend(['Train acc.','Validation Acc.'],fontsize=15)
ax.set_xlabel('Epochs',size=15)

ax=fig.add_subplot(1,3,3)
plot_decision_regions(X=x_valid,y=y_valid.astype(np.integer),
                      clf=model)
ax.set_xlabel(r'$x_1$',size=15)
ax.xaxis.set_label_coords(1,-0.025)
ax.set_ylabel(r'$x_2$',size=15)
ax.yaxis.set_label_coords(-0.025,1)
plt.show()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(
    units=4,
    input_shape = (2,),
    activation = 'relu'))
model.add(tf.keras.layers.Dense(
    units=4,
    activation = 'relu'))
model.add(tf.keras.layers.Dense(
    units=4,
    activation = 'relu'))
model.add(tf.keras.layers.Dense(
    units=1,
    activation = 'sigmoid'))
model.summary()

model.compile (optimizer = tf.keras.optimizers.SGD(),
               loss = tf.keras.losses.BinaryCrossentropy(),
               metrics=[tf.keras.metrics.BinaryAccuracy()])
hist = model.fit(x_train,y_train,
                 validation_data = (x_valid,y_valid),
                 epochs=200, batch_size = 2, verbose=0)

history = hist.history
fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(1,3,1)
plt.plot(history['loss'],lw=4)
plt.plot(history['val_loss'],lw=4)
plt.legend(['Train loss','Validation loss'],fontsize=15)
ax.set_xlabel('Epochs',size=15)

ax=fig.add_subplot(1,3,2)
plt.plot(history['binary_accuracy'],lw=4)
plt.plot(history['val_binary_accuracy'],lw=4)
plt.legend(['Train acc.','Validation Acc.'],fontsize=15)
ax.set_xlabel('Epochs',size=15)

ax=fig.add_subplot(1,3,3)
plot_decision_regions(X=x_valid,y=y_valid.astype(np.integer),
                      clf=model)
ax.set_xlabel(r'$x_1$',size=15)
ax.xaxis.set_label_coords(1,-0.025)
ax.set_ylabel(r'$x_2$',size=15)
ax.yaxis.set_label_coords(-0.025,1)
plt.show()

# Making model building more flexible with Keras' function API

tf.random.set_seed(1)

## input layer:
inputs = tf.keras.Input(shape=(2,))

## hidden layers
h1 = tf.keras.layers.Dense(units=4, activation='relu')(inputs)
h2 = tf.keras.layers.Dense(units=4, activation='relu')(h1)
h3 = tf.keras.layers.Dense(units=4, activation='relu')(h2)

## output:
outputs = tf.keras.layers.Dense(units=1, activation='sigmoid')(h3)

## construct a model:
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()
    
## compile :
model.compile(
    optimizer=tf.keras.optimizers.SGD(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy()])

## train:
hist = model.fit(
    x_train,y_train,
    validation_data=(x_valid,y_valid),
    epochs=200, verbose=0,batch_size=2)


history = hist.history
fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(1,3,1)
plt.plot(history['loss'],lw=4)
plt.plot(history['val_loss'],lw=4)
plt.legend(['Train loss','Validation loss'],fontsize=15)
ax.set_xlabel('Epochs',size=15)

ax=fig.add_subplot(1,3,2)
plt.plot(history['binary_accuracy'],lw=4)
plt.plot(history['val_binary_accuracy'],lw=4)
plt.legend(['Train acc.','Validation Acc.'],fontsize=15)
ax.set_xlabel('Epochs',size=15)

ax=fig.add_subplot(1,3,3)
plot_decision_regions(X=x_valid,y=y_valid.astype(np.integer),
                      clf=model)
ax.set_xlabel(r'$x_1$',size=15)
ax.xaxis.set_label_coords(1,-0.025)
ax.set_ylabel(r'$x_2$',size=15)
ax.yaxis.set_label_coords(-0.025,1)
plt.show()

# Implementing models based on Keras' Model class

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.hidden_1 = tf.keras.layers.Dense(
            units=4, activation='relu')
        self.hidden_2 = tf.keras.layers.Dense(
            units=4, activation='relu')
        self.hidden_3 = tf.keras.layers.Dense(
            units=4, activation='relu')
        self.output_layer = tf.keras.layers.Dense(
            units=1, activation = 'sigmoid')
    def call(self,input):
        h = self.hidden_1(input)
        h = self.hidden_2(h)
        h = self.hidden_3(h)
        return self.output_layer(h)
    
tf.random.set_seed(1)
model = MyModel()
model.build(input_shape=(None,2))

model.summary()

## compile:
model.compile(optimizer = tf.keras.optimizers.SGD(),
              loss = tf.keras.losses.BinaryCrossentropy(),
              metrics = [tf.keras.metrics.BinaryAccuracy()])

hist = model.fit(x_train,y_train,
                 validation_data=(x_valid,y_valid),
                 epochs = 200, batch_size = 2, verbose=0)

history = hist.history
fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(1,3,1)
plt.plot(history['loss'],lw=4)
plt.plot(history['val_loss'],lw=4)
plt.legend(['Train loss','Validation loss'],fontsize=15)
ax.set_xlabel('Epochs',size=15)

ax=fig.add_subplot(1,3,2)
plt.plot(history['binary_accuracy'],lw=4)
plt.plot(history['val_binary_accuracy'],lw=4)
plt.legend(['Train acc.','Validation Acc.'],fontsize=15)
ax.set_xlabel('Epochs',size=15)

ax=fig.add_subplot(1,3,3)
plot_decision_regions(X=x_valid,y=y_valid.astype(np.integer),
                      clf=model)
ax.set_xlabel(r'$x_1$',size=15)
ax.xaxis.set_label_coords(1,-0.025)
ax.set_ylabel(r'$x_2$',size=15)
ax.yaxis.set_label_coords(-0.025,1)
plt.show()

# writing custom Keras layers
class NoisyLinear(tf.keras.layers.Layer):
    def __init__(self,output_dim,noise_stddev=0.1,**kwargs):
        self.output_dim = output_dim
        self.noise_stddev = noise_stddev
        super(NoisyLinear,self).__init__(**kwargs)
        
    def build(self,input_shape):
        self.w = self.add_weight(name='weights',
                                 shape=(input_shape[1],
                                        self.output_dim),
                                 initializer='random_normal',
                                 trainable = True)
        self.b = self.add_weight(shape=(self.output_dim,),
                                 initializer = 'zeros',
                                 trainable = True)
    def call(self, inputs, training=False):
        if training:
            batch = tf.shape(inputs)[0]
            dim = tf.shape(inputs)[1]
            noise = tf.random.normal(shape=(batch,dim),
                                     mean=0.0,
                                     stddev=self.noise_stddev)
            noisy_inputs = tf.add(inputs, noise)
        else:
            noisy_inputs = inputs
        z = tf.matmul(noisy_inputs,self.w)+self.b
        return tf.keras.activations.relu(z)
    
    def get_config(self):
        config = super(NoisyLinear, self).get_config()
        config.update({'output_dim': self.output_dim,
                       'noise_stddev':self.noise_stddev})
        return config
    
#test the new layer
tf.random.set_seed(1)
noisy_layer = NoisyLinear(4)
noisy_layer.build(input_shape=(None, 4))
x =tf.zeros(shape=(1,4))
tf.print(noisy_layer(x,training=True))

## re-building from config:
config = noisy_layer.get_config()
new_layer = NoisyLinear.from_config(config)
tf.print(new_layer(x, training = True))

#using the new defined layer in XOR classification model
tf.random.set_seed(1)
model = tf.keras.Sequential([
    NoisyLinear(4,noise_stddev=0.1),
    tf.keras.layers.Dense(units=4,activation = 'relu'),
    tf.keras.layers.Dense(units = 4, activation = 'relu'),
    tf.keras.layers.Dense(units = 1, activation ='sigmoid')])

model.build(input_shape=(None,2))
model.summary()

## compile:
model.compile(
    optimizer = tf.keras.optimizers.SGD(),
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics =[tf.keras.metrics.BinaryAccuracy()])

##training:
hist= model.fit(x_train,y_train,
                validation_data=(x_valid,y_valid),
                epochs=200, batch_size=2, verbose=0)

history = hist.history
fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(1,3,1)
plt.plot(history['loss'],lw=4)
plt.plot(history['val_loss'],lw=4)
plt.legend(['Train loss','Validation loss'],fontsize=15)
ax.set_xlabel('Epochs',size=15)

ax=fig.add_subplot(1,3,2)
plt.plot(history['binary_accuracy'],lw=4)
plt.plot(history['val_binary_accuracy'],lw=4)
plt.legend(['Train acc.','Validation Acc.'],fontsize=15)
ax.set_xlabel('Epochs',size=15)

ax=fig.add_subplot(1,3,3)
plot_decision_regions(X=x_valid,y=y_valid.astype(np.integer),
                      clf=model)
ax.set_xlabel(r'$x_1$',size=15)
ax.xaxis.set_label_coords(1,-0.025)
ax.set_ylabel(r'$x_2$',size=15)
ax.yaxis.set_label_coords(-0.025,1)
plt.show()

# TensorFlow Estimators

import pandas as pd
dataset_path = tf.keras.utils.get_file("auto-mpg.data",
                                       ("http://archive.ics.uci.edu/ml/machine-learning"
                                        "-databases/auto-mpg/auto-mpg.data"))
column_names =[
    'MPG', 'Cylinders', 'Displacement',
    'Horsepower', 'Weight', 'Acceleration',
    'ModelYear','Origin']

df=pd.read_csv(dataset_path,names=column_names,
               na_values='?', comment='\t',
               sep=' ',skipinitialspace=True)

## drop the NA rows
df=df.dropna()
df = df.reset_index(drop=True)

#train/test split:
import sklearn 
import sklearn.model_selection

df_train,df_test=sklearn.model_selection.train_test_split(df,train_size =0.8)
train_stats=df_train.describe().transpose()

numeric_column_names = [
    'Cylinders', 'Displacement',
    'Horsepower','Weight',
    'Acceleration']

df_train_norm,df_test_norm=df_train.copy(),df_test.copy()

for col_name in numeric_column_names:
    mean = train_stats.loc[col_name,'mean']
    std = train_stats.loc[col_name,'std']
    df_train_norm.loc[:,col_name]=(
        df_train_norm.loc[:,col_name]-mean)/std
    df_test_norm.loc[:,col_name]=(
        df_test_norm.loc[:,col_name]-mean)/std
df_train_norm.tail()

numeric_features=[]
for col_name in numeric_column_names:
    numeric_features.append(
        tf.feature_column.numeric_column(key=col_name))

feature_year = tf.feature_column.numeric_column(key='ModelYear')
bucketized_features=[]
bucketized_features.append(tf.feature_column.bucketized_column(
    source_column=feature_year,
    boundaries=[73,76,79]))

feature_origin = tf.feature_column.categorical_column_with_vocabulary_list(
    key='Origin',
    vocabulary_list=[1,2,3])

categorical_indicator_features=[]
categorical_indicator_features.append(
    tf.feature_column.indicator_column(feature_origin))

##### Machine Learning with pre-made Estimators
#Step1 : Define an input function for data loading
def train_input_fn(df_train,batch_size=8):
    df= df_train.copy()
    train_x,train_y = df,df.pop('MPG')
    dataset = tf.data.Dataset.from_tensor_slices(
        (dict(train_x),train_y))
    #shuffle, repeat, and batch the examples
    return dataset.shuffle(1000).repeat().batch(batch_size)

 #load a batch to see how does it look like
ds = train_input_fn(df_train_norm)
batch = next(iter(ds))
print('Keys:', batch[0].keys())

print('Batch Model Years:', batch[0]['ModelYear'])   

## define an input function for the test dataset that will be used for evaluation after model training
def eval_input_fn(df_test,batch_size=8):
    df = df_test.copy()
    test_x,test_y=df,df.pop('MPG')
    dataset = tf.data.Dataset.from_tensor_slices(
        (dict(test_x),test_y))
    return dataset.batch(batch_size)

#Step2 : Convert the dataset into feature columns
all_feature_columns =(
    numeric_features+
    bucketized_features+
    categorical_indicator_features)

#Step3: Instantiate an Estimator 
regressor = tf.estimator.DNNRegressor(feature_columns=all_feature_columns,
                                      hidden_units=[32,10],
                                      model_dir='models/autompg-dnnregressor/')

#Step4: Use the Estimator methods train(), evaluate(), and predict()

EPOCHS = 1000
BATCH_SIZE = 8
total_steps  = EPOCHS * int(np.ceil(len(df_train)/BATCH_SIZE))
print('Training Steps:', total_steps)

regressor.train(input_fn=lambda:train_input_fn(
    df_train_norm,batch_size=BATCH_SIZE),
    steps=total_steps)

# reload the last checkpoint
reloaded_regressor = tf.estimator.DNNRegressor(hidden_units=[32,10], 
                                             feature_columns=all_feature_columns,
                                             warm_start_from ='models/autompg-dnnregressor/')

# Evaluate the predictive performance of the trained model
eval_results = reloaded_regressor.evaluate(
    input_fn=lambda:eval_input_fn(df_test_norm,batch_size=8))

print('Average-Loss {:.4f}'.format(
    eval_results['average_loss']))

#predict the target valueson new points
pred_res = regressor.predict(
    input_fn=lambda:eval_input_fn(df_test_norm, batch_size=8))

print(next(iter(pred_res)))

