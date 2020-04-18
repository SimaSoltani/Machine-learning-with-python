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
