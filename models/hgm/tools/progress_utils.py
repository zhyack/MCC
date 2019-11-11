#coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import codecs

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = codecs.open(filename, 'a', 'UTF-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

from tqdm import tqdm

class Bar():
    def __init__(self, description, total_num, messages=None):
        self.bar = tqdm(desc=description, total=total_num, postfix=messages, ncols=80, bar_format='{desc}: {bar} [{n_fmt}/{total_fmt}] {elapsed}<{remaining} {postfix}')

    def end(self, message=None):
        self.bar.close()
        if message:
            print(message)
    
    def update(self, escaped_num=1, message=None):
        self.bar.update(n=escaped_num)
        self.bar.set_postfix(ordered_dict=message)
    
class LossInfo():
    def __init__(self, cats):
        self.losses = {}
        self.last_losses = {}
        for cat in cats:
            self.losses[cat] = {}
            self.last_losses[cat] = {}
    def push(self, cat, losses):
        if not (isinstance(losses, tuple) or isinstance(losses, list)):
            losses = [losses]
        for ind, loss in enumerate(losses):
            if ind not in self.losses[cat]:
                self.losses[cat][ind] = [-1, 0]
                self.last_losses[cat][ind] = -1
            self.losses[cat][ind][0] = (self.losses[cat][ind][0]*self.losses[cat][ind][1]+loss) / (self.losses[cat][ind][1]+1)
            self.losses[cat][ind][1] += 1
            self.last_losses[cat][ind] = loss
    def info(self):
        message = ""
        for cat in self.losses:
            message += 'Loss_%s: '%(cat.upper())
            if len(self.losses[cat])>1:
                for i in self.losses[cat]:
                    message += '(%d) %.3f '%(i, self.losses[cat][i][0])
            else:
                for i in self.losses[cat]:
                    message += '%.3f '%(self.losses[cat][i][0])
        return message
    def lastInfo(self):
        message = ""
        for cat in self.last_losses:
            message += 'Loss_%s: '%(cat.upper())
            if len(self.last_losses[cat])>1:
                for i in self.losses[cat]:
                    message += '(%d) %.3f '%(i, self.last_losses[cat][i])
            else:
                for i in self.losses[cat]:
                    message += '%.3f '%(self.last_losses[cat][i])
        return message
