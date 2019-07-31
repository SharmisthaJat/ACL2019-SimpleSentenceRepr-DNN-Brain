#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Author: Sharmistha Jat and Hao Tang
# Paper: ACL 2019: Relating Simple Sentence Representations in Deep Neural Networks and the Brain

from model_cuda import *

from batcher_multi_task import *
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


def train_per_step(train_batcher, model, criterion, alpha, optimiser):
    
    train_batcher.shuffle()
    step_par_epoch = train_batcher.max_batch_num
    t_total = 0
    for j in range(step_par_epoch):

        [X_context, target_pos, target_nw] = train_batcher.next_whole_seq()
        target_pos = torch.LongTensor(target_pos).reshape(train_batcher.batch_size, -1)
        target_nw = torch.LongTensor(target_nw).reshape(train_batcher.batch_size, -1)

        for k in range(1, len(X_context[0]) + 1):
            context_data = Variable(torch.LongTensor(X_context)[:,:k].cuda())
            context_target_pos = Variable(target_pos[:,k-1].cuda())
            context_target_nw = Variable(target_nw[:,k-1].cuda())
            _, _, y_pred_pos, y_pred_nw, context_whole_seq, context_per_seq = model(context_data, whole_seq=False)
            
            loss_pos = criterion(y_pred_pos, context_target_pos)
            loss_nw = criterion(y_pred_nw, context_target_nw)

            # process multiple words of the decoded output and calculate the loss
            loss_svo=0
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss_svo += criterion(decoder_output, target_variable[di])
                decoder_input = target_variable[di]

            loss = loss_pos + loss_nw + loss_svo
            t_total += loss.data.item()
            
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            gc.collect()
    return str(t_total / (train_batcher.max_pad*step_par_epoch))
        
# train_whole_step(train_batcher, model, criterion, optimiser, logFile)    
def train_whole_step(train_batcher, model, criterion, alpha, optimiser):
    train_batcher.shuffle()
    step_par_epoch = train_batcher.max_batch_num
    t_total = 0
    
    for j in range(step_par_epoch):


        [X_context, target_pos, target_nw] = train_batcher.next_whole_seq()

        context_data = Variable(torch.LongTensor(X_context).cuda())
        context_target_pos = Variable(torch.LongTensor(target_pos).cuda())
        context_target_nw = Variable(torch.LongTensor(target_nw).cuda())
        
        _, _, y_pred_pos, y_pred_nw = model(context_data, whole_seq=True)
        y_pred_pos = y_pred_pos.reshape(y_pred_pos.shape[0] * y_pred_pos.shape[1], -1)
        y_pred_nw = y_pred_nw.reshape(y_pred_nw.shape[0] * y_pred_nw.shape[1], -1)

        loss_pos = criterion(y_pred_pos, context_target_pos)
        loss_nw = criterion(y_pred_nw, context_target_nw)

        loss = alpha*loss_pos + (1-alpha)*loss_nw
        t_total += loss.data.item()
        
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        gc.collect()
        
    return str(t_total / step_par_epoch)

# evaluate_loss(dev_batcher, model, scheduler, logFile, eval_best)
def evaluate_loss(dev_batcher, model, criterion, output_path, description):
    total = 0
    dev_batcher.shuffle()
    step_par_epoch = dev_batcher.max_batch_num
    for j in range(step_par_epoch):

        [X_context, target_pos, target_nw] = dev_batcher.next_whole_seq()

        context_data = Variable(torch.LongTensor(X_context).cuda())
        context_target_pos = Variable(torch.LongTensor(target_pos).cuda())
        context_target_nw = Variable(torch.LongTensor(target_nw).cuda())

        _, _, y_pred_pos, y_pred_nw = model(context_data, whole_seq=True)
        y_pred_pos = y_pred_pos.reshape(y_pred_pos.shape[0] * y_pred_pos.shape[1], -1)
        y_pred_nw = y_pred_nw.reshape(y_pred_nw.shape[0] * y_pred_nw.shape[1], -1)

        loss_pos = criterion(y_pred_pos, context_target_pos)
        loss_nw = criterion(y_pred_nw, context_target_nw)

        loss = loss_pos + loss_nw
        total += loss.data.item()

 
    return total / step_par_epoch

def evaluate_acc(dev_batcher, model, criterion, output_path, description):
    
    total, acc = 0, 0
    dev_batcher.shuffle()
    step_par_epoch = dev_batcher.max_batch_num
    
    for j in range(step_par_epoch):
        
        [X_context, target_pos, _] = dev_batcher.next_whole_seq()
        context_data = Variable(torch.LongTensor(X_context).cuda())
        context_target = torch.LongTensor(target_pos).numpy()
        _, _, y_pred_pos, _  = model(context_data, whole_seq=True)
        y_pred_pos = np.argmax(y_pred_pos.cpu().data.view(
                y_pred_pos.shape[0] * y_pred_pos.shape[1], -1).numpy(), axis=1)
        acc += np.sum(y_pred_pos == context_target)
        total += y_pred_pos.shape[0]
                

    return acc, total

def evaluate_acc_per_step(dev_batcher, model, criterion, output_path, description):
    
    total, acc = 0, 0
    dev_batcher.shuffle()
    step_par_epoch = dev_batcher.max_batch_num
    
    for j in range(step_par_epoch):
        
        [X_context, target_data, _] = dev_batcher.next_whole_seq()
        target_data = torch.LongTensor(target_data).reshape(dev_batcher.batch_size, -1)

        for k in range(1, len(X_context[0]) + 1):
            context_data = Variable(torch.LongTensor(X_context)[:,:k].cuda())
            context_target = torch.LongTensor(target_data[:,k-1]).numpy()
            _, _, y_pred, _  = model(context_data, whole_seq=False)
            y_pred = np.argmax(y_pred.cpu().data.numpy(), axis=1)
            
            
            acc += np.sum(y_pred == context_target)
            total += dev_batcher.batch_size
                
    return acc, total

def main(args):
    
    data_X = pickle.load(open(args.data_X))
    data_y = pickle.load(open(args.data_y))
    id2vec = pickle.load(open("./preprocessed_data/index_to_vector_X.p"))
    tag2word = pickle.load(open("./preprocessed_data/index_to_word_y.p"))
    
    random.seed(args.seed)
    train, dev = partition(zip(data_X, data_y), args.small)
    train_X, train_y = zip(*train)
    dev_X, dev_y = zip(*dev)
    batch_size = args.batch_size 
    emb_dim = args.emb_size
    hidden_size = int(args.h_size)
    
    kwargs = {}
    kwargs['output_size'] = len(tag2word)
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
    
    if args.version == 'v1':
        model = MultiTaskModel_v1(kwargs).cuda()
    else:
        model = MultiTaskModel_v2(kwargs).cuda()
    model.initGloveEmb(id2vec_params)
        
    #def __init__(self,X,y,batch_size,num_of_samples,id2vec,pad,id):
    # Might have a better abstraction to write this
    
    # Create batcher
    ##############################################################
    
    
    train_batcher = MultiBatcher(train_X, train_y, batch_size, 
                                 len(train_X), args.max_pad)
    
    dev_batcher = MultiBatcher(dev_X, dev_y, batch_size, 
                                 len(dev_X), args.max_pad)
    
    ##############################################################

    
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimiser = torch.optim.SGD(model.parameters(), lr=1)    
    scheduler = s.StepLR(optimiser, step_size=5, gamma=0.5)
    scheduler.step()
    logFile = open(args.output_path + description + '.txt', 'w')
    eval_best = None    
    
    hyper_param = vars(args)
    print(hyper_param)
    logFile.write(str(hyper_param))
    for i in range(int(args.max_epoch)):
        
        if args.train_per_seq:
            loss = train_per_step(train_batcher, model, criterion, args.alpha, optimiser)
        else:
            loss = train_whole_step(train_batcher, model, criterion, args.alpha, optimiser)
            
    
        logFile.write('train loss:' + loss +'\n')
        print('Train: ', i, ' Loss: ', loss)
    
        if args.evaluate_metric == 'loss':
            eval_loss = evaluate_loss(dev_batcher, model, criterion, args.output_path, description)

            if not eval_best or eval_loss < eval_best:
                with open(args.output_path + "multi_task.model_" + args.description, 'w') as f:
                    torch.save(model,f)
                    eval_best = eval_loss

            logFile.write('Eval loss:' + str(eval_loss) + '\n')
            print('Eval: ', i, ' Loss: ', str(eval_loss))
            
        else:
            if args.train_per_seq:
                acc, total = evaluate_acc_per_step(dev_batcher, model, criterion, args.output_path, description)
            else:
                acc, total = evaluate_acc(dev_batcher, model, criterion, args.output_path, description)
                
            if not eval_best or acc >= eval_best:
                with open(args.output_path + "multi_task.model_" + args.description, 'w') as f:
                    torch.save(model,f)
                    eval_best = acc
                    
            logFile.write('Eval acc:' + str(acc) + '/' + str(total) +'\n')
            print('Eval acc:' + str(acc) + '/' + str(total) +'\n')
        
        scheduler.step()

    logFile.close()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout", type=float, help="dropout", default=0.0)
    parser.add_argument("--max_epoch", help="max_epoch")
    parser.add_argument("--max_pad", type=int, help="max padding", default=15)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=20)
    parser.add_argument("--h_size", type=int, help="hidden_size", default=300)
    parser.add_argument("--alpha", type=float, help="ratio loss of pos to nw", default=0.8)
    parser.add_argument("--emb_size", type=int, help="embedding_size", default=300)
    parser.add_argument("--data_X", help="path to data_X.", default='./preprocessed_data/wiki_nell_X.p')
    parser.add_argument("--data_y", help="path to data_y.", default='./preprocessed_data/wiki_nell_y.p')
    parser.add_argument("--output_path", help="output path of best model and logfile", default='./result_multi/')
    parser.add_argument("--tied", help="tied", action='store_true')
    parser.add_argument("--bidir", help="is bidirectional", action='store_true')
    parser.add_argument("--train_per_seq", help="training one step a time", action='store_true')
    parser.add_argument("--evaluate_metric", help="choosing how to evaluate the model", choices=['loss', 'acc'])
    parser.add_argument("--version", help="choosing the version of model", choices=['v1', 'v2'])
    parser.add_argument("--small", help="trained on small data", action='store_true')
    parser.add_argument("--seed", type=int, help="seed of randomization", default=0)
    parser.add_argument("--description", default='baseline')
    args = parser.parse_args()
    
    
    main(args)
