#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import itertools


class MultiBatcher(object):
    
    # In a multi-task setting, X and Y should be list of lists
    def __init__(self, X, y, batch_size, num_samples, max_pad):
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.max_batch_num = num_samples / batch_size
        self.batch_num = 0
        self.max_pad = max_pad
        self.X, self.y_pos, self.y_nw = self.preprocess_data(X, y)
     
    def truncate_or_pad(self, sample):
        if len(sample) > self.max_pad:
            sample = sample[: self.max_pad]
        else:
            zero_padding = [0] * max(0, self.max_pad - len(sample))
            sample += zero_padding
        return sample
        
    def preprocess_data(self, X, y):
        X_group = [[self.truncate_or_pad(X[c * self.batch_size + x]) for x in range(self.batch_size)]\
                    for c in range(self.max_batch_num)]
        
        y_pos = [list(itertools.chain.from_iterable([self.truncate_or_pad(y[c * self.batch_size + x]) for x in range(self.batch_size)]))\
                    for c in range(self.max_batch_num)]
        
        y_nw = [list(itertools.chain.from_iterable([batch[i][1:] + [0] for i in range(len(batch))])) for batch in X_group]
        return X_group, y_pos, y_nw
        
    def next_whole_seq(self):
        cur_index = self.batch_num
        self.batch_num = (self.batch_num + 1) % self.max_batch_num
        return self.X[cur_index], self.y_pos[cur_index], self.y_nw[cur_index]

    def shuffle(self):
        combined = zip(self.X, self.y_pos, self.y_nw)
        np.random.shuffle(combined)
        self.X, self.y_pos, self.y_nw = zip(*combined)


