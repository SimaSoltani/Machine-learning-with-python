# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 21:50:02 2020

@author: Sima Soltani
"""

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

#split train, validation and test sets
tf.random.set_seed(1)
ds_raw = ds_raw.shuffle(
    50000,reshuffle_each_iteration = False)
ds_raw_test = ds_raw.take(25000)
ds_raw_train_valid = ds_raw.skip(25000)
ds_raw_train = ds_raw_train_valid.take(20000)
ds_raw_valid = ds_raw_train_valid.skip(20000)

from collections import Counter
def preprocess_datasets(
        ds_raw_train,
        ds_raw_valid,
        ds_raw_test,
        max_seq_length=None,
        batch_size=32):
    ##( step 1 is already done)
    ##Step2: find unique tokens
    tokenizer = tfds.features.text.Tokenizer()
    token_counts = Counter()
    
    for example in ds_raw_train:
        tokens = tokenizer.tokenize(example[0].numpy()[0])
        if max_seq_length is not None:
            tokens = tokens[-max_seq_length:]
            token_counts.update(tokens)
            
    print('Vocab-size:', len(token_counts))
    
    ##Step3: encoding the texts
    encoder = tfds.features.text.TokenTextEncoder(
        token_counts)
    
    def encode(text_tensor, label):
        text = text_tensor.numpy()[0]
        encoded_text = encoder.encode(text)
        if max_seq_length is not None:
            encoded_text[-max_seq_length:]
        return encoded_text,label
    
    def encode_map_fn(text,label):
        return tf.py_function(encode,inp=[text,label],
                              Tout=(tf.int64,tf.int64))
    
    ds_train = ds_raw_train.map(encode_map_fn)
    ds_valid =ds_raw_valid.map(encode_map_fn)
    ds_test = ds_raw_test.map(encode_map_fn)
    
    ##Step 4: batching the datasets
    train_data = ds_train.padded_batch(batch_size,padded_shapes=([-1],[]))
    valid_data = ds_valid.padded_batch(batch_size,padded_shapes=([-1],[]))
    test_data = ds_test.padded_batch(batch_size,padded_shapes=([-1],[]))
    
    
    return (train_data,valid_data,test_data,len(token_counts))

from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense

def build_rnn_model(embedding_dim,vocab_size,
                    recurrent_type='SimpleRNN',
                    n_recurrent_units=64,
                    n_recurrent_layers=1,
                    bidirectional=True):
    tf.random.set_seed(1)
    
    #build the model
    model= tf.keras.Sequential()
    
    model.add(
        Embedding(
            input_dim = vocab_size,
            output_dim=embedding_dim,
            name = 'embed-layer'))
    
    for i in range(n_recurrent_layers):
        return_sequences = (i<n_recurrent_layers-1)
        
        if recurrent_type =='SimpleRNN':
            recurren_layer = SimpleRNN(
                units=n_recurrent_units,
                return_sequences = return_sequences,
                name = 'Simprnn-layer-{}'.format(i))
        elif recurrent_type =='LSTM':
            recurren_layer = LSTM(
                units=n_recurrent_units,
                retuen_sequences = return_sequences,
                name = 'lstm-layer-{}'.format(i))
        elif recurrent_type =='GRU':
            recurren_layer = GRU(
                units=n_recurrent_units,
                retuen_sequences=return_sequences,
                name = 'GRU-layer-{}'.format(i))
        if bidirectional:
            recurren_layer= Bidirectional(
                recurren_layer,name='bidir-'+recurren_layer.name)
        model.add(recurren_layer)
        
    model.add(Dense(64,activation ='relu'))
    model.add(Dense(1,activation = 'sigmoid'))
    
    return model

batch_size = 32
embedding_dim = 20 
max_seq_length = 100

train_data,valid_data,test_data, n = preprocess_datasets(
    ds_raw_train,ds_raw_valid,ds_raw_test,max_seq_length=max_seq_length,
    batch_size=batch_size
    )

vocab_size = n+2

rnn_model = build_rnn_model(
    embedding_dim,vocab_size,
    recurrent_type='SimpleRNN',
    n_recurrent_units=64,
    n_recurrent_layers=1,
    bidirectional=True)

rnn_model.summary()

rnn_model.compile(
    optimizer = tf.keras.optimizers.Adam(1e-3),
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=['accuracy'])

history = rnn_model.fit(train_data,validation_data=valid_data,epochs=10)
