#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np



class nwBatcher:
    
    def __init__(self, X, batch_size, num_of_samples, id2vec, max_pad):
        self.X = X
        self.num_of_samples = num_of_samples
        self.batch_size = batch_size
        self.max_batch_num = int(self.num_of_samples / self.batch_size)
        self.id2vec = id2vec
        self.batch_num = 0
        self.max_pad = max_pad
        self.X_seq, self.X_cur = None, []
        self.y_seq, self.y_cur = None, []
        self.cur_step = 0

    # Get the batch with complete sequence
    def next_whole_seq(self):

        X_context = self.X[self.batch_num * self.batch_size : (self.batch_num + 1) * self.batch_size]
        X_target = []
        for i in range(len(X_context)):
            if len(X_context[i]) > self.max_pad:
                X_context[i] = X_context[i][: self.max_pad]
            
            else:
                zero_padding = [0] * max(0, self.max_pad - len(X_context[i]))
                X_context[i].extend(zero_padding)
            
            if len(X_context[i]) > self.max_pad:
                raise Exception('Batcher error, did not get the right length')
                
            X_target.extend(X_context[i][1:] + [0])
            
            

        self.batch_num = (self.batch_num + 1) % self.max_batch_num
        return X_context, X_target

    def next_single_seq(self):
        if self.cur_step == 0:
            self.X_seq, self.y_seq = self.next_whole_seq()
            self.X_cur, self.y_cur = self.X_seq[:, 0], self.y_seq[:, 0]

    def shuffle(self):
        combined = self.X
        np.random.shuffle(combined)
        self.X = combined


