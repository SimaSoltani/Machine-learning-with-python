# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:22:43 2020

@author: Sima Soltani
"""

import tensorflow as tf
tf.random.set_seed(1)

rnn_layer =tf.keras.layers.SimpleRNN(
    units =2, 
    use_bias = True,
    return_sequences = True)
rnn_layer.build(input_shape=(None,None,5))

w_xh,w_oo,b_h = rnn_layer.weights
print('W_xh shape:',w_xh.shape)
print('W_oo shape: ',w_oo.shape)
print('b_h shape: ',b_h.shape)

#call the forward pass on the rnn_layer and manually compute the outputs 
#at each time step and compare them 

x_seq = tf.convert_to_tensor(
    [[1.0]*5,[2.0]*5,[3.0]*5],dtype=tf.float32)

##output of SimpleRNN:
output = rnn_layer(tf.reshape(x_seq, shape=(1,3,5)))


##manually computing the output:
out_man =[]
for t in range (len(x_seq)):
    xt = tf.reshape(x_seq[t],(1,5))
    print('Time step {}=>'.format(t))
    print('     Input        :',xt.numpy())
    
    ht = tf.matmul(xt,w_xh)+b_h
    print('       hidden     :', ht.numpy())
    
    if t>0:
        prev_o = out_man[t-1]
    else:
        prev_o = tf.zeros(shape=(ht.shape))
    ot = ht +tf.matmul(prev_o, w_oo)
    ot = tf.math.tanh(ot)
    out_man.append(ot)
    print('    Output (manual):',ot.numpy())
    print('    SimpleRNN output:'.format(t),
          output[0][t].numpy())
    print()

    

#Project One - predicting the sentiment if the IMDb movies reviews
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd

df = pd.read_csv('movie_data.csv',encoding='utf-8')

## step 1: create a dataset
target = df.pop('sentiment')
ds_raw = tf.data.Dataset.from_tensor_slices(
    (df.values,target.values))

## inspection:
for ex in ds_raw.take(3):
    tf.print(ex[0].numpy()[0][:50],ex[1])
#split train, validation and test sets
tf.random.set_seed(1)
ds_raw = ds_raw.shuffle(
    50000,reshuffle_each_iteration = False)
ds_raw_test = ds_raw.take(25000)
ds_raw_train_valid = ds_raw.skip(25000)
ds_raw_train = ds_raw_train_valid.take(20000)
ds_raw_valid = ds_raw_train_valid.skip(20000)

##step2 :find unique tokens(words)
from collections import Counter

tokenizer = tfds.features.text.Tokenizer()
token_counts = Counter()

for example in ds_raw_train:
    tokens = tokenizer.tokenize(example[0].numpy() [0])
    token_counts.update(tokens)
    
print('Vocab_size:', len(token_counts))

##step3 : encoding unique tokens to integers
encoder = tfds.features.text.TokenTextEncoder(token_counts)
example_str ='This is an example!'
print(encoder.encode(example_str))

#Step3-A: Define the function for transformation
def encode(text_tensor,label):
    text = text_tensor.numpy()[0]
    encoded_text = encoder.encode(text)
    return encoded_text,label

#Step3-b: wrap the encode function to a TF Op.
def encode_map_fn(text,label):
    return tf.py_function(encode,inp=[text,label],
                          Tout=(tf.int64,tf.int64))


ds_train = ds_raw_train.map(encode_map_fn)
ds_test = ds_raw_test.map(encode_map_fn)
ds_valid = ds_raw_valid.map(encode_map_fn)

# look at the ashape of some examples
tf.random.set_seed(1)
for example in ds_train.shuffle(1000).take(5):
     print('Sequence length:',example[0].shape)
     
## take a small subset
ds_subset = ds_train.take(8)
for example in ds_subset:
    print('Individual size:',example[0].shape)
    
##Dividing the dataset into batches
ds_batched =ds_subset.padded_batch(
    4,padded_shapes=([-1],[]))

for batch in ds_batched:
    print('Batch dimention:',batch[0].shape)
    
#devide all the three datsets into mini batches
train_data = ds_train.padded_batch(
    32,padded_shapes=([-1],[]))

test_data = ds_test.padded_batch(
    32,padded_shapes=([-1],[]))

valid_data = ds_valid.padded_batch(
    32,padded_shapes=([-1],[]))

