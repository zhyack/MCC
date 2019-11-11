#coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import codecs

from blue_print import TrainPlan, TestPlan
from hgm.tools.progress_utils import Logger

from hgm.model import HGModel
from hgm.tools.jsonrw import json2load, save2json

import argparse
parser = argparse.ArgumentParser(
    description="Specify the config files!")
parser.add_argument("-c", dest="config_name", type=str, default=None, help="The preset config file name (under configs/). ")
parser.add_argument("-plan", dest="train_or_test", type=str, default=None, help="config: train_or_test")
parser.add_argument("-load", dest="load_folder_file", type=str, default=None, help="config: load_model & load_folder_file")
parser.add_argument("-save", dest="save_folder", type=str, default=None, help="config: save_model & save_folder")
parser.add_argument("-opt", dest="optimizer", type=str, default=None, help="config: model.optimizer")
parser.add_argument("-lr", dest="lr", type=float, default=None, help="config: model.lr")
parser.add_argument("-norm", dest="clip_norm", type=float, default=None, help="config: model.clip_norm")
parser.add_argument("-bs", dest="bs_width", type=int, default=None, help="config: model.use_bs & model.bs_width")
parser.add_argument("-iters", dest="iters", type=int, default=None, help="config: plan.iters")
parser.add_argument("-routi", dest="iter_routing", type=int, default=None, help="config: plan.iters")


def fixArgs(d, args):
    if args.iter_routing:
        d['model']['iter_routing']=args.iter_routing
    if args.train_or_test:
        d['train_or_test'] = args.train_or_test
    if args.load_folder_file:
        d['load_model'] = True
        d['load_folder_file'] = os.path.split(args.load_folder_file)
    if args.save_folder:
        d['save_model'] = True
        d['save_folder_file'] = args.save_folder
    if args.optimizer:
        d['model']['optimizer'] = args.optimizer
    if args.lr:
        d['model']['lr'] = args.lr
    if args.clip_norm:
        d['model']['clip_norm'] = args.clip_norm
    if args.bs_width!=None:
        if args.bs_width>0:
            d['model']['use_bs'] = True
            d['model']['bs_width'] = args.bs_width
        else:
            d['model']['use_bs'] = False
    if args.iters:
        d['plan']['iters'] = args.iters
    
    if d['train_or_test'] not in ['train', 'test', 'final_test']:
        raise Exception("CONFIG $train_or_test should be `train` or `test`")
    d['plan']['save_folder'] = d['save_folder']
    d['plan']['batch_size'] = d['model']['batch_size']
    if d['train_or_test']=='test':
        d['load_model'] = True
        if not os.path.exists(d['load_folder_file'][0]):
            raise Exception("Load Folder does not exist!")
        d['save_model'] = False
        d['plan']['iters'] = 1
        d['plan']['save_test'] = True
        d['plan']['load_checkpoint_name'] = d['load_folder_file'][1]
        d['plan']['load_folder_file'] = d['load_folder_file']
        d['model']['is_train'] = False
        d['model']['encoder_dropout']=1.0
        d['model']['decoder_dropout']=1.0
        d['model']['caps_dropout']=0.0
    else:
        d['model']['use_bs'] = False
    
    if d['save_model']==False:
        #d['plan']['save_folder'] = None
        d['plan']['save_valid'] = False
    
    d['plan']['batch_size'] = d['model']['batch_size']
    return d

if __name__=="__main__":
    given_args = parser.parse_args()
    args = json2load('configs/default.json')
    args = {}
    if given_args.config_name:
        specified_args = json2load('configs/%s.json'%(given_args.config_name))
        args.update(specified_args)

    if args['load_model'] and os.path.exists(os.path.join(args['load_folder_file'][0], 'config.json')):
        load_args = json2load(os.path.join(args['load_folder_file'][0], 'config.json'))
        args.update(load_args)
        
    args = fixArgs(args, given_args)
    # print('Got configurations from %s:\n'%(given_args.config_name), args)
    model = RGModel(args['model'])

    if args['load_model']:
        model.load_checkpoint(args['load_folder_file'][0], args['load_folder_file'][1])
    
    if args['save_model']:
        if not os.path.exists(args['save_folder']):
            print("Save Folder does not exist! Making directory %s"%format(args['save_folder']))
            os.mkdir(args['save_folder'])
        if not os.path.exists(os.path.join(args['save_folder'], 'log.txt')):
            f = codecs.open(os.path.join(args['save_folder'], 'log.txt'), 'w', 'UTF-8')
            f.close()
        save2json(args, os.path.join(args['save_folder'], 'config.json'))
        sys.stdout = Logger(os.path.join(args['save_folder'], 'log.txt'), sys.stdout)
        sys.stderr = Logger(os.path.join(args['save_folder'], 'log.txt'), sys.stderr)
    
    if args['train_or_test'].lower() == 'train':
        big_plan = TrainPlan(model, args['plan'])
    else:
        big_plan = TestPlan(model, args['plan'])
    big_plan.execute()

