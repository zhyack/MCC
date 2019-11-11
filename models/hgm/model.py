#coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np

from .net import HGNet
from .model_utils import load_checkpoint, save_checkpoint
from .data_utils import loadDict, getEncoderData, getDecoderData

class HGModel():
    def __init__(self, args):
        self.args = args
        
        self.text_dict, self.text_list = loadDict(self.args['dict_file'])
        self.args['dict_size'] = len(self.text_list)
        self.args['id_eos'] = self.text_dict['<EOS>']
        self.net = RGNet(self.args)

        tfconfig = tf.ConfigProto(gpu_options=tf.GPUOptions())
        tfconfig.gpu_options.allow_growth=True
        self.sess = tf.Session(graph=self.net.graph, config=tfconfig)
        self.saver = None
        with tf.Session() as temp_sess:
            temp_sess.run(tf.global_variables_initializer())
        self.sess.run(tf.variables_initializer(self.net.graph.get_collection('variables')))

        self.train_funcs = self.train_gen
        self.predict_funcs = self.predict_gen
    
    def train_gen(self, data_batch):
        xs, ys = list(zip(*data_batch))

        return loss

    def predict_gen(self, data_batch, require_loss=False):
        xs, ys = list(zip(*data_batch))

        if require_loss:
            return (results, loss)
        else:
            return results
    
    
    
    def train(self, data_batch):
        return self.train_funcs(data_batch)
    def valid(self, data_batch):
        return self.predict_funcs(data_batch, require_loss=True)
    def test(self, data_batch):
        return self.predict_funcs(data_batch)
    
    def load_checkpoint(self, folder, filename):
        load_checkpoint(self, folder, filename)
        
    def save_checkpoint(self, folder, filename):
        save_checkpoint(self, folder, filename)