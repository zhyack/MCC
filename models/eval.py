#coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import codecs

import argparse
parser = argparse.ArgumentParser(
    description="Specify the files!")
parser.add_argument("-gold", dest="pf_gold", type=str, default=None)
parser.add_argument("-pred", dest="pf_pred", type=str, default=None)

given_args = parser.parse_args()

def readData(pfin, pfout):  
    data_in=[]
    data_out=[]
    f = codecs.open(pfin, 'r', 'UTF-8')
    inlines = f.readlines()
    f.close()
    f = codecs.open(pfout, 'r', 'UTF-8')
    outlines = f.readlines()
    f.close()
    for si, so in zip(inlines, outlines):
        data_in.append(si.strip())
        data_out.append(so.strip())
    return data_in, data_out

golds, preds = readData(given_args.pf_gold, given_args.pf_pred)

import hgm.metrics as metrics

print('BLEU: ', metrics.all_funcs['BLEU'](golds, preds))
print('PRF: ', metrics.all_funcs['PRF'](golds, preds))
print('WER: ', metrics.all_funcs['WER'](golds, preds))
print('DIST1: ', metrics.all_funcs['DIST1'](golds, preds))
print('DIST2: ', metrics.all_funcs['DIST2'](golds, preds))
print('DIST3: ', metrics.all_funcs['DIST3'](golds, preds))

