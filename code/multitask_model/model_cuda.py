#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Author: Sharmistha Jat and Hao Tang
# Paper: ACL 2019: Relating Simple Sentence Representations in Deep Neural Networks and the Brain

import torch
import torch.nn as nn

from torch.autograd import Variable


# Shared embedding layers and LSTM layer. Only differnt in linear layer
class MultiTaskModel_v1(nn.Module):
    def __init__(self,kwargs):
        super(MultiTaskModel_v1, self).__init__()
        self.emb_dim = kwargs['emb_dim']
        self.vocab_size = kwargs['vocab_size']
        self.hidden_size = kwargs['hidden_size']
        self.output_size = kwargs['output_size']
        self.n_layers = kwargs['num_layers']
        self.droprate = kwargs['dropout']
        self.bidir = kwargs['bidir']
        self.linearPred_pos = nn.Linear(self.hidden_size, self.output_size)
        self.linearPred_nw = nn.Linear(self.hidden_size, self.vocab_size)
        self.drop = nn.Dropout(self.droprate)
        self.multi = 2 if self.bidir else 1


        ##change emb to hidden
        self.lstm = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_size, num_layers=self.n_layers, \
                            bidirectional=self.bidir,dropout=self.droprate,batch_first=True)
        nn.init.xavier_uniform_(self.linearPred_pos.weight)
        nn.init.xavier_uniform_(self.linearPred_nw.weight)

        self.glove_embeddings = nn.Embedding(self.vocab_size, self.emb_dim )
        #self.glove_embeddings.requires_grad=False
        if kwargs['tied']:
            if self.emb_dim != self.hidden_size:
                raise ValueError('emb size not equal to hid size')
            self.linearPred_pos.weight = self.glove_embeddings.weight 
            self.linearPred_nw.weight = self.glove_embeddings.weight 
            
    def initGloveEmb(self, init_emb):
        if self.glove_embeddings.weight.shape[1] == 300:
            print('Initialize with glove Embeddings')
            self.glove_embeddings.weight = nn.Parameter(torch.from_numpy(init_emb).float().cuda())


    def initHidden(self, N):
        ##change emb to hidden
        multi = self.multi
        return (Variable(torch.randn(multi*self.n_layers, N, self.hidden_size).zero_().cuda()),
                Variable(torch.randn(multi*self.n_layers, N, self.hidden_size).zero_().cuda()))

    def forward(self, context_input, whole_seq=True):
        context = self.drop(self.glove_embeddings(context_input))
        init_hidd = self.initHidden(context_input.size()[0])
        context_whole_seq, context_per_seq = self.lstm(context, init_hidd)
        
        ##batch,seqlen,hidden
        context_m = context_whole_seq if whole_seq else context_per_seq[0].mean(0)
        context_m1 = context_m
        
        # Normally, Training whole sequence with bidirectional lstm will not happen.
        
        if whole_seq and self.bidir:
            context_m1 = context_m.view(context_m.shape[0], context_m.shape[1], self.multi, -1)
            context_m1 = context_m1.mean(2)
            
        context_d = self.drop(context_m1)
        context_d = context_d.contiguous()
        
        # Expandable here if whoe seq, then will be b*s*h, else will be b*h
        
        context_pos = self.linearPred_pos(context_d)
        context_nw = self.linearPred_nw(context_d)
        
        return context, context_m, context_pos, context_nw
    
    
    
# Shared LSTM layer. Different in embedding layer and linear layer.
class MultiTaskModel_v2(nn.Module):
    def __init__(self,kwargs):
        super(MultiTaskModel_v2, self).__init__()
        self.emb_dim = kwargs['emb_dim']
        self.vocab_size = kwargs['vocab_size']
        self.hidden_size = kwargs['hidden_size']
        self.output_size = kwargs['output_size']
        self.n_layers = kwargs['num_layers']
        self.droprate = kwargs['dropout']
        self.bidir = kwargs['bidir']

        self.drop = nn.Dropout(self.droprate)
        self.multi = 2 if self.bidir else 1


        ##change emb to hidden
        self.lstm = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_size, num_layers=self.n_layers, \
                            bidirectional=self.bidir,dropout=self.droprate,batch_first=True)


        # If there is different vocab_size here, need to output two vocab size here
        self.glove_embeddings_pos = nn.Embedding(self.vocab_size, self.emb_dim)
        self.glove_embeddings_nw = nn.Embedding(self.vocab_size, self.emb_dim)
        
        self.linearPred_pos = nn.Linear(self.hidden_size, self.output_size)
        self.linearPred_nw = nn.Linear(self.hidden_size, self.vocab_size)

        nn.init.xavier_uniform_(self.linearPred_pos.weight)
        nn.init.xavier_uniform_(self.linearPred_nw.weight)
        #self.glove_embeddings.requires_grad=False
        if kwargs['tied']:
            if self.emb_dim != self.hidden_size:
                raise ValueError('emb size not equal to hid size')
            self.linearPred_pos.weight = self.glove_embeddings_pos.weight 
            self.linearPred_nw.weight = self.glove_embeddings_nw.weight 

            
    def initGloveEmb(self, init_emb):
        if self.glove_embeddings_pos.weight.shape[1] == 300:
            print('Initialize task pos tagging with glove Embeddings')
            self.glove_embeddings_pos.weight = nn.Parameter(torch.from_numpy(init_emb).float().cuda())

        if self.glove_embeddings_nw.weight.shape[1] == 300:
            print('Initialize task langauge model with glove Embeddings')
            self.glove_embeddings_nw.weight = nn.Parameter(torch.from_numpy(init_emb).float().cuda())


    def initHidden(self, N):
        ##change emb to hidden
        multi = self.multi
        return (Variable(torch.randn(multi*self.n_layers, N, self.hidden_size).zero_().cuda()),
                Variable(torch.randn(multi*self.n_layers, N, self.hidden_size).zero_().cuda()))

    def forward(self, context_input, whole_seq=True):
        
        emb = self.glove_embeddings_pos(context_input)
        emb = self.glove_embeddings_nw(context_input)
        
        context = self.drop(emb)
        init_hidd = self.initHidden(context_input.size()[0])
        context_whole_seq, context_per_seq = self.lstm(context, init_hidd)
        
        ##batch,seqlen,hidden
        context_m = context_whole_seq if whole_seq else context_per_seq[0].mean(0)
        context_m1 = context_m
        
        if whole_seq and self.bidir:
            context_m1 = context_m.view(context_m.shape[0], context_m.shape[1], self.multi, -1)
            context_m1 = context_m1.mean(2)
        context_d = self.drop(context_m1)
        context_d = context_d.contiguous()
        
        # Expandable here if whoe seq, then will be b*s*h, else will be b*h
        
        context_pos = self.linearPred_pos(context_d)
        context_nw = self.linearPred_nw(context_d)

        return context, context_m, context_pos, context_nw
