#coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np

from .model_utils import Dense, updateBP
from .seq_utils import modelInitWordEmbedding, modelGetWordEmbedding, modelInitBidirectionalEncoder, modelRunBidirectionalEncoder, modelInitHierarchicalAttentionDecoderCell, modelRunDecoderForTrain, modelRunDecoderForGreedyInfer, modelRunDecoderForBSInfer
from .caps_utils import *

class HGNet():
    def __init__(self, args):
        self.args = args
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            # define graph here!

            self.all_variables = tf.trainable_variables()
            if args['is_train']:
                optimizer = tf.train.AdamOptimizer
                if self.args['optimizer']=='GD':
                    optimizer = tf.train.GradientDescentOptimizer
                self.train_op = updateBP(self.final_loss, [self.args['lr']], [tf.trainable_variables()], optimizer, norm=self.args['clip_norm'])
            
            

