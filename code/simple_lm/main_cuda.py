#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Author: Sharmistha Jat and Hao Tang
# Paper: ACL 2019: Relating Simple Sentence Representations in Deep Neural Networks and the Brain

from model_cuda import * ##change to no_cuda
from batcher_next_word import *
import cPickle as pickle
import torch
import torch.optim.lr_scheduler as s
from torch.autograd import Variable
import numpy as np
import gc
import argparse
import math
import random 
from random import shuffle

def partition(data, test_small):
    shuffle(data)        
    data_len = len(data)
    a = int(math.ceil(0.9 * data_len))
    
    if test_small:
        a = int(math.ceil(0.01 * data_len))
        b = int(math.ceil(0.99 * data_len))
        return data[:a], data[b:]
    return data[:a], data[a:]

def train_per_step(train_batcher, model, criterion, optimiser):
    
    train_batcher.shuffle()
    step_par_epoch = train_batcher.max_batch_num
    max_pad = train_batcher.max_pad
    t_total = 0
    for j in range(step_par_epoch):

        [X_context, target_data] = train_batcher.next_whole_seq()
        target_data = torch.LongTensor(target_data).reshape(train_batcher.batch_size, -1)
        for k in range(1, len(X_context[0]) + 1):
            context_data = Variable(torch.LongTensor(X_context)[:,:k].cuda())
            context_target = Variable(target_data[:,k-1].cuda())

            _, _, y_pred_batch = model(context_data, whole_seq=False)
            
            loss = criterion(y_pred_batch, context_target)
            t_total += loss.data.item()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            gc.collect()
    return str(t_total / (max_pad*step_par_epoch))
        
# train_whole_step(train_batcher, model, criterion, optimiser, logFile)    
def train_whole_step(train_batcher, model, criterion, optimiser):
    train_batcher.shuffle()
    step_par_epoch = train_batcher.max_batch_num
    t_total = 0
    for j in range(step_par_epoch):

        [X_context, target_data] = train_batcher.next_whole_seq()    
        

        context_data = Variable(torch.LongTensor(X_context).cuda())
        context_target = Variable(torch.LongTensor(target_data).cuda())
        
        _, _, y_pred_batch = model(context_data, whole_seq=True)
        y_pred_batch = y_pred_batch.reshape(y_pred_batch.shape[0] * y_pred_batch.shape[1], -1)
        loss = criterion(y_pred_batch, context_target)
        t_total += loss.data.item()
        
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        gc.collect()
        
    return str(t_total / step_par_epoch)

# evaluate_loss(dev_batcher, model, scheduler, logFile, eval_best)
# For language model. We use perplexity as our evaluation
def evaluate_loss(dev_batcher, model, criterion, output_path, description):
    total = 0
    dev_batcher.shuffle()
    step_par_epoch = dev_batcher.max_batch_num
    for j in range(step_par_epoch):
        [X_context, X_target] = dev_batcher.next_whole_seq()

        context_data = Variable(torch.LongTensor(X_context).cuda())
        context_target = Variable(torch.LongTensor(X_target).cuda())            

        _, _, y_pred_batch  = model(context_data)
        y_pred_batch = y_pred_batch.reshape(y_pred_batch.shape[0] * y_pred_batch.shape[1], -1)
        loss = criterion(y_pred_batch, context_target)
        total += loss.data.item()
 
    return total / step_par_epoch

def evaluate_loss_per_step(dev_batcher, model, criterion, output_path, description):
    
    total = 0
    dev_batcher.shuffle()
    step_par_epoch = dev_batcher.max_batch_num
    max_pad = dev_batcher.max_pad
    
    for j in range(step_par_epoch):
        
        [X_context, target_data] = dev_batcher.next_whole_seq()
        target_data = torch.LongTensor(target_data).reshape(dev_batcher.batch_size, -1)

        for k in range(1, len(X_context[0]) + 1):
            context_data = Variable(torch.LongTensor(X_context)[:,:k].cuda())
            context_target = torch.LongTensor(target_data[:,k-1]).numpy()
            _, _, y_pred  = model(context_data, whole_seq=False)
            loss = criterion(y_pred, context_target)            
            
            total += loss.data.item()
                
    return total / (max_pad*step_par_epoch)

def main(args):
    
    data = pickle.load(open(args.data_path))
    id2vec = pickle.load(open("./preprocessed_data/index_to_vector_X.p"))
    
    random.seed(args.seed)
    train, dev = partition(data, args.small)
    batch_size = args.batch_size 
    emb_dim = args.emb_size
    hidden_size = int(args.h_size)
    
    kwargs = {}
    kwargs['bidir'] = args.bidir
    kwargs['dropout'] = args.dropout
    kwargs['tied'] = args.tied
    kwargs['num_layers'] = 2
    kwargs['batch_size'] = batch_size
    kwargs['vocab_size'] = len(id2vec)
    kwargs['hidden_size'] = hidden_size
    kwargs['emb_dim'] = emb_dim
    description = args.description
    
    # embeddings in numpy form
    id2vec_params = []
    for i in range(max(id2vec.keys()) + 1):
        try:
            id2vec_params.append(id2vec[i])
        except:
            print 'err in id2vec'
    
    
    assert len(id2vec) == len(id2vec_params), 'id2vec_param len wrong'
    
    id2vec_params = np.array(id2vec_params)
    
    
    if args.model_name == 'lstm_nw':
        model = LSTMNextWordModel(kwargs).cuda()
        model.initGloveEmb(id2vec_params)
        
        # def __init__(self, X, batch_size, num_of_samples, id2vec, max_pad)
        train_batcher = nwBatcher(train, batch_size, len(train), id2vec, args.max_pad)
        dev_batcher = nwBatcher(dev, batch_size, len(dev), id2vec, args.max_pad)
    else:
        raise ValueError('No other choices beside lstm yet')
    
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimiser = torch.optim.SGD(model.parameters(), lr=20)
    
    scheduler = s.StepLR(optimiser, step_size=1, gamma=0.25)
    logFile = open(args.output_path + description + '.txt', 'w')
    eval_best = None    
    
    hyper_param = vars(args)
    print(hyper_param)
    logFile.write(str(hyper_param) + '\n')
    for i in range(int(args.max_epoch)):
        
        if args.train_per_seq:
            loss = train_per_step(train_batcher, model, criterion, optimiser)
        else:
            loss = train_whole_step(train_batcher, model, criterion, optimiser)
            
    
        logFile.write('train loss:' + loss +'\n')
        print('Train: ', i, ' Loss: ', loss)
    
        # dev
        if args.train_per_seq:
            eval_loss = evaluate_loss_per_step(dev_batcher, model, criterion, 
                                               args.output_path, description)
        else:
            eval_loss = evaluate_loss(dev_batcher, model, criterion, args.output_path, description)            
        perplexity = math.exp(eval_loss)
        if not eval_best or perplexity < eval_best:
            with open(args.output_path + "nw.model_" + args.description, 'w') as f:
                torch.save(model,f)
                eval_best = perplexity
        else:
            scheduler.step()  
        logFile.write('Eval perplexity:' + str(perplexity) +'\n')
        print('Eval perplexity:' + str(perplexity) +'\n')

    logFile.close()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="model_name", choices=["lstm_nw"])
    parser.add_argument("--dropout", type=float, help="dropout", default=0.5)
    parser.add_argument("--max_epoch", help="max_epoch")
    parser.add_argument("--max_pad", type=int, help="max padding", default=15)
    parser.add_argument("--seed", type=int, help="seed of randomization", default=0)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=20)
    parser.add_argument("--h_size", type=int, help="hidden_size", default=100)
    parser.add_argument("--emb_size", type=int, help="embedding_size", default=100)
    parser.add_argument("--data_path", help="path to data. No need for label", default='./preprocessed_data/wiki_nell_X.p')
    parser.add_argument("--output_path", help="output path of best model and logfile", default='./result_nw/')
    parser.add_argument("--small", help="trained on small data", action='store_true')
    parser.add_argument("--tied", help="tied", action='store_true')
    parser.add_argument("--bidir", help="is bidirectional", action='store_true')
    parser.add_argument("--train_per_seq", help="training one step a time", action='store_true')
    parser.add_argument("--description", default='baseline')
    args = parser.parse_args()
    
    
    main(args)
    
