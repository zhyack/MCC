#coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score 
import numpy as np
from .tools.bleu import *


def NormalBleu(gold, predicted):
    mbleu, addition = corpus_bleu(predicted, [g.split('\t') for g in gold])
    return "BLEU = %.2f, %.1f/%.1f/%.1f/%.1f (BP=%.3f, ratio=%.3f, hyp_len=%d, ref_len=%d)"%(mbleu[0]*100, mbleu[1]*100, mbleu[2]*100, mbleu[3]*100, mbleu[4]*100, addition[0], addition[1], addition[2], addition[3])

def Acc(gold,predicted):
    gold_int=[int(i) for i in gold]
    predicted_int=[int(j) for j in predicted]
    acc = accuracy_score(gold_int, predicted_int)
    print ( 'Acc: %.3f' % acc)  

def Precision_Recall_F1(gold,predicted):
    #gold: 1d array-like Ground truth (correct) target values
    #predicted: 1d array-like Estimated targets as returned by a classifier
    [precision, recall, F1, support] = \
        precision_recall_fscore_support(gold,predicted, average='samples')
    print ('Precision: %.3f' % precision, 'Recall: %.3f' % recall, 'F1: %.3f' % F1)      
#its hard to design metrics for clustering
def Given_items(gold_items,predicted):
    recall=[]
    for i in range(len(predicted)):
        input_ings=gold_items[i]
        cnt_tmp=0
        for j in gold_items[i]:
            if input_ings[j] in gold_items[i]:
                cnt_tmp+=1
        recall.append(float(cnt_tmp)/float(len(input_ings)))
    print ('Given_items:%.3f',np.mean(np.array(recall)))

def ngramCnt(t, max_n):
    s = set()
    cnt = 0
    wl = t.split()
    n = len(wl)
    for i in range(n-max_n+1):
        ngram = ' '.join(wl[i:i+max_n-1])
        s.add(ngram)
        cnt += 1
    return s, cnt


def corpus_diversity(texts, max_n=2):
    total_s = set()
    total_cnt = 0
    for t in texts:
        s, cnt = ngramCnt(t, max_n)
        total_s = total_s.union(s)
        total_cnt += cnt
    if total_cnt == 0:
        return 0.0
    else:
        return len(total_s)/float(total_cnt)
def DIST1(golds, predicts):
    return corpus_diversity(predicts, max_n=1)
def DIST2(golds, predicts):
    return corpus_diversity(predicts, max_n=2)
def DIST3(golds, predicts):
    return corpus_diversity(predicts, max_n=3)

import os
import codecs
from .data_utils import save2text

def meteor_evaluate(gold, predicts):
    n = 1
    if gold[0].find('\t')!=-1:
        refs = [g.split('\t') for g in gold]
        refs = list(zip(*refs))
        n = len(refs)
        for i, ref in enumerate(refs):
            save2text(ref, 'tmp/ref-%d.txt'%(i))
    else:
        save2text(gold, 'tmp/ref-0.txt')
    save2text(predicts, 'tmp/pred.txt')
    
    pf_ref = ""
    for i in range(n):
        pf_ref += " somewhere/tmp/ref-%d.txt"%(i)
    os.popen('cd wherethefollwingfileis/; python tools/make_meteor_file.py -i %s -o %s' %(pf_ref, "somewhere/tmp/ref.meteor"))

    
    outputs = ''.join(os.popen('cd wherethefollwingfileis/;java -Xmx2G -jar tools/meteor-1.5/meteor-1.5.jar somewhere/tmp/pred.txt somewhere/tmp/ref.meteor -l en -norm -r %d' % (n)).readlines()[-1:])
    return outputs

def wer(r, h):
    """
    This is a function that calculate the word error rate in ASR.
    You can use it like this: wer("what is it".split(), "what is".split()) 
    """
    #build the matrix
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0: d[0][j] = j
            elif j == 0: d[i][0] = i
    for i in range(1,len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    result = float(d[len(r)][len(h)]) / max(len(r), len(h)) * 100
    # result = str("%.2f" % result) + "%"
    return result

def ASR(gold, predicts):
    ret = 0
    n = len(gold)
    for g, p in zip(gold, predicts):
        ret += wer(g.split(), p.split())
    if (n==0):
        return 0.0
    else:
        return ret/n

from collections import Counter
def prf_score(gold_items, pred_items):
    """
    Computes precision, recall and f1 given a set of gold and prediction items.
    :param pred_items: iterable of predicted values
    :param gold_items: iterable of gold values
    :return: tuple (p, r, f1) for precision, recall, f1
    """
    common = Counter(gold_items) & Counter(pred_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1
def PRF(gold, predicts):
    ret_p = 0.0
    ret_r = 0.0
    ret_f = 0.0
    n = len(gold)
    for g, p in zip(gold, predicts):
        p, r, f  = prf_score(g.split(), p.split())
        ret_p += p
        ret_r += r
        ret_f += f
    if (n!=0):
        ret_p /= n
        ret_r /= n
        ret_f /= n
    return 'Precision: %.6f, Recall: %.6f, F1: %.6f'%(ret_p, ret_r, ret_f)



all_funcs = {'BLEU': NormalBleu,'Acc':Acc, 'DIST1':DIST1, 'DIST2':DIST2, 'DIST3':DIST3,'meteor':meteor_evaluate, 'WER':ASR, 'PRF':PRF}
