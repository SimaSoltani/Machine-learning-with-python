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
