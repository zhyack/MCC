#coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np
import copy
from tensorflow.python.layers import core as layers_core
import math
import os

def Dense(size, use_bias=False):
    return layers_core.Dense(size, use_bias=use_bias)

def updateBP(loss, lr, var_list, optimizer, norm=None):
    gradients = [tf.gradients(loss, var_list[i]) for i in range(len(lr))]
    if norm!=None:
        gradients = [tf.clip_by_global_norm(gradients[i], norm)[0] for i in range(len(lr))]
    optimizers = [optimizer(lr[i]) for i in range(len(lr))]
    return [optimizers[i].apply_gradients(zip(gradients[i], var_list[i])) for i in range(len(lr))]

def save_checkpoint(model, folder='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(folder, filename)
    if not os.path.exists(folder):
        print("Checkpoint Directory does not exist! Making directory {}".format(folder))
        os.mkdir(folder)
    else:
        print("Checkpoint Directory exists! ")
    if model.saver == None:
        model.saver = tf.train.Saver(model.net.graph.get_collection('variables'), max_to_keep=50)
    with model.net.graph.as_default():
        model.saver.save(model.sess, filepath)

def load_checkpoint(model, folder='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath+'.meta'):
        print("No model in path {}".format(filepath))
        return
    with model.net.graph.as_default():
        print("Find model in path {}".format(filepath))
        print('Restoring Models...')

        try:
            model.saver = tf.train.Saver(model.net.all_variables)
            model.saver.restore(model.sess, filepath)
            print('Model Loaded!')
        except Exception as e:
            print(e)
            print('Model Not Found Or Failed...')