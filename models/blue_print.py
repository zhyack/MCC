#coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import codecs

import hgm.metrics as metrics
from hgm.data_utils import getData, rearrangeBatch, save2text, gatherPredictedInfo
from hgm.tools.jsonrw import json2load, save2json
from hgm.tools.progress_utils import Bar, LossInfo

import time
import random
import json


class TrainPlan():
    def __init__(self, model, args):
        self.model = model
        self.args = args
    
    def train(self):
        if 'train_sample_n' in self.args['data']:
            sxs = self.args['data']['train_sample_n']
        else:
            sxs = 0
        all_train_data = getData(sxs, self.args['data']['train_in'], self.args['data']['train_out'])
        all_train_data = rearrangeBatch(all_train_data, self.args['batch_size'], use_shuffle=True)
        
        cat='yy'
        losses = LossInfo([cat])
        bar = Bar("%s Train"%(cat.upper()), len(all_train_data))
        message = None
        for d in all_train_data:
            loss = self.model.train(d)
            losses.push(cat, loss)
            message = {'INFO':losses.info()}
            bar.update(1, message)
        bar.end("%s Training Finished with losses: "%(cat.upper()) + losses.lastInfo())
        
    def valid(self):
        if 'valid_sample_n' in self.args['data']:
            sxs = self.args['data']['valid_sample_n']
        else:
            sxs = 0
        all_valid_data = getData(sxs, self.args['data']['valid_in'], self.args['data']['valid_out'])
        ns = len(all_valid_data)
        all_valid_data = rearrangeBatch(all_valid_data, self.args['batch_size'], use_shuffle=False)
        
        cat='yy'
    

        
        losses = LossInfo([cat])
        bar = Bar("%s Valid"%(cat.upper()), len(all_valid_data))
        message = None
        inputs = []
        gold = []
        predicted = []
        for d in all_valid_data:
            outputs, loss = self.model.valid(d)
            inp, g, pred = gatherPredictedInfo(d, outputs)
            inputs.extend(inp)
            gold.extend(g)
            predicted.extend(pred)

            losses.push(cat, loss)
            message = {'INFO':losses.info()}
            bar.update(1, message)
        bar.end("%s Validating Finished with losses: "%(cat.upper())+ losses.lastInfo())
        for m in self.args['metrics']:
            mfunc = metrics.all_funcs[m]
            results = mfunc(gold[:ns], predicted[:ns])
            print('%s: '%(m), results)
        semantic_weights, persona_weights = self.model.getPrintableWeights()
        print('Semantic Weights:', semantic_weights)
        print('Persona Weights:', persona_weights)
        if self.args['save_valid']:
            # save2text(inputs, os.path.join(self.args['save_folder'], 'valid_inputs.txt'))
            save2text(gold, os.path.join(self.args['save_folder'], 'valid_gold.txt'))
            save2text(predicted, os.path.join(self.args['save_folder'], 'valid_predicted_%d.txt'%(self.iter_i)))

    def execute(self):
        for self.iter_i in range(self.args['iters']):
            print('Iter %d begin...'%(self.iter_i))
            self.train()
            self.valid()
            if self.args['save_folder']:
                self.model.save_checkpoint(self.args['save_folder'], 'new')
                self.model.save_checkpoint(self.args['save_folder'], '%d'%(self.iter_i))
                print('Model saved @ %s'%(self.args['save_folder']))

class TestPlan():
    def __init__(self, model, args):
        self.model = model
        self.args = args

    def test(self):
        if 'test_sample_n' in self.args['data']:
            sxs = self.args['data']['test_sample_n']
        else:
            sxs = 0
        all_test_data = getData(sxs, self.args['data']['test_in'], self.args['data']['test_out'])
        ns = len(all_test_data)
        all_test_data = rearrangeBatch(all_test_data, self.args['batch_size'], use_shuffle=False)
        
        cat='yy'

        
        losses = LossInfo([cat])
        bar = Bar("%s Test"%(cat.upper()), len(all_test_data))
        message = None
        inputs = []
        gold = []
        predicted = []
        for d in all_test_data:
            outputs = self.model.test(d)
            inp, g, pred = gatherPredictedInfo(d, outputs)
            inputs.extend(inp)
            gold.extend(g)
            predicted.extend(pred)
            bar.update(1, message)
        bar.end("Test Finished.")
        for m in self.args['metrics']:
            mfunc = metrics.all_funcs[m]
            results = mfunc(gold[:ns], predicted[:ns])
            print('%s: '%(m), results)
        if self.args['save_test']:
            #save2text(inputs, os.path.join(self.args['save_folder'], 'test_inputs.txt'))
            save2text(gold, os.path.join(self.args['save_folder'], 'test_gold.txt'))
            save2text(predicted, os.path.join(self.args['save_folder'], 'test_predicted_%s.txt'%(self.args['load_folder_file'][1])))


    def execute(self):
        self.test()
