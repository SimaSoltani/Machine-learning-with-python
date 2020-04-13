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
