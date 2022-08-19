from textwrap import indent
import numpy as np
import torch
from typing import List, Tuple, Set
import pickle
import torch.optim as optim
import torch.nn as nn
import random
import collections
from termcolor import colored
import pdb
import functools
import pandas as pd
from config import PAD, Config
from datastruct import Span

def get_labelids(config):
    df_train = pd.read_json('../data/'+config.dataset+'/'+config.encoder+'_train.json', lines=True)
    df_dev = pd.read_json('../data/'+config.dataset+'/'+config.encoder+'_dev.json', lines=True)
    df_test = pd.read_json('../data/'+config.dataset+'/'+config.encoder+'_test.json', lines=True)
    labels_train = [sub_l for l in df_train["labels"] for sub_l in l] + [sub_l for l in df_train["gold_labels"] for sub_l in l]
    labels_dev = [sub_l for l in df_dev["labels"] for sub_l in l] + [sub_l for l in df_dev["gold_labels"] for sub_l in l]
    labels_test = [sub_l for l in df_test["labels"] for sub_l in l] + [sub_l for l in df_test["gold_labels"] for sub_l in l]
    unique_labels = pd.unique(labels_train+labels_dev+labels_test).tolist()
    unique_labels.sort()

    label2idx = {l: i+2 for i, l in enumerate(unique_labels)}
    idx2label = [config.PAD, config.UNK] + [l for l in unique_labels]

    label2idx[config.PAD] = 0
    label2idx[config.UNK] = 1

    config.label2idx = label2idx
    config.idx2label = idx2label
    
    config.label2idx[config.START_TAG] = len(config.label2idx)
    config.idx2label.append(config.START_TAG)
    config.label2idx[config.STOP_TAG] = len(config.label2idx)
    config.idx2label.append(config.STOP_TAG)
    config.label_size = len(config.label2idx)
    config.start_label_id = config.label2idx[config.START_TAG]
    config.stop_label_id = config.label2idx[config.STOP_TAG]

    if config.cutoff=='fake':
        config.label2idx['fakepos'] = len(config.label2idx)
        config.idx2label.append('fakepos')
        config.label2idx['fakeneg'] = len(config.label2idx)
        config.idx2label.append('fakeneg')

    config.label_size = len(config.label2idx)    

def create_folds(config, insts, folds):
    # sentence entities: only the entities from each utterance
    # folds: number of folds
    sentence_entities = [[e for e in insts[i]['labels'] if e!='O'] for i in range(len(insts))]
    data_size = len(sentence_entities)
    fold_size = int(np.ceil(data_size/folds))
    indexs = list(range(data_size))
    allind = []
    info = {'seed': config.seed, 'folds': folds, 'indexs': indexs}
    random.shuffle(indexs)
    for i in range(folds):
        test_data_indexs = indexs[(i*fold_size):int(min((i+1)*fold_size, data_size))]
        allind.extend(test_data_indexs)
        train_data_indexs = [indexs[x] for x in range(data_size) if x not in list(range((i*fold_size),int(min((i+1)*fold_size, data_size))))]
        
        # entity joint filtering, to avoid overfitting
        # forbid_entities = set().union(*[set(sentence_entities[x]) for x in test_data_indexs])
        # train_data_indexs = list(
        #     filter(lambda x: set(sentence_entities[x]).isdisjoint(forbid_entities), train_data_indexs))
        # assert set(test_data_indexs).isdisjoint(set(train_data_indexs))
        # assert set().union(*[set(sentence_entities[x]) for x in test_data_indexs]).isdisjoint(
        #     set().union(*[set(sentence_entities[x]) for x in train_data_indexs]))
        _info = {
            'train_indexs': train_data_indexs,
            'test_indexs': test_data_indexs,
            'train_sentences': len(train_data_indexs),
            'train_total_entities': sum(len(sentence_entities[x]) for x in train_data_indexs),
            'train_distinct_entities': len(set().union(*[set(sentence_entities[x]) for x in train_data_indexs])),
            'test_sentences': len(test_data_indexs),
            'test_total_entities': sum(len(sentence_entities[x]) for x in test_data_indexs),
            'test_distinct_entities': len(set().union(*[set(sentence_entities[x]) for x in test_data_indexs])),
        }

        info[f'fold-{i}'] = _info
        print(f"Set {i}")
        print(f"Train sentences: {_info['train_sentences']}")
        print(f"Train total entities: {_info['train_total_entities']}")
        print(f"Train distinct entities: {_info['train_distinct_entities']}")
        print(f"Test sentences: {_info['test_sentences']}")
        print(f"Test total entities: {_info['test_total_entities']}")
        print(f"Test distinct entities: {_info['test_distinct_entities']}")
 
    assert all([x == y for x,y in zip(allind,indexs)])
    return info

def gen_dic(labels,label2idx):
    types=set()
    for label in labels:
        if(label.startswith('B') or label.startswith('S') or label.startswith('E') or label.startswith('I')):
            tp=label.split('-')[1]
            types.add(tp)
    pos_dic={'O':[label2idx['O']]}
    type_dic={'O':[label2idx['O']]}
    for label in labels:
        if(label=='O' or label.startswith('<') or label.startswith('fake')):
            continue
        pos,type=label.split('-')[0],label.split('-')[1]
        if(pos in pos_dic):
            pos_dic[pos].append(label2idx[label])
        else:
            pos_dic[pos]=[label2idx[label]]
        if(type in type_dic):
            type_dic[type].append(label2idx[label])
        else:
            type_dic[type]=[label2idx[label]]
    for tp in types:
        type_dic[tp].append(label2idx['O'])
    for pos in ['B','I','E','S']:
        pos_dic[pos].append(label2idx['O'])
    return pos_dic,type_dic



def gen_embedding_table(idx2label,type_dic,pos_dic):
    type_embedding=torch.zeros(len(idx2label),len(idx2label))
    pos_embedding=torch.zeros(len(idx2label),len(idx2label))
    #type_embedding
    for id,label in enumerate(idx2label):
        
        if(label.startswith('B') or label.startswith('S') or label.startswith('E') or label.startswith('I')):
            indexes=type_dic[label.split('-')[1]]
            for index in indexes:
                type_embedding[id][index]=1
        elif(label=='O'):
            type_embedding[id]=torch.ones_like(type_embedding[id])
            
    #pos_embedding
    for id,label in enumerate(idx2label):
        
        if(label.startswith('B') or label.startswith('S') or label.startswith('E') or label.startswith('I')):
            indexes=pos_dic[label.split('-')[0]]
            for index in indexes:
                pos_embedding[id][index]=1
        elif(label=='O'):
            pos_embedding[id]=torch.ones_like(pos_embedding[id])
            
    type_embedding,pos_embedding =pos_embedding,type_embedding
    return type_embedding,pos_embedding


def flatten(t):
    if isinstance(t[0],list):
        return functools.reduce(lambda x,y: x+y,t)
    else:
        return t    


def log_sum_exp_pytorch(vec: torch.Tensor) -> torch.Tensor:
    """
    Calculate the log_sum_exp trick for the tensor.
    :param vec: [batchSize * from_label * to_label].
    :return: [batchSize * to_label]
    """
    maxScores, idx = torch.max(vec, 1)
    maxScores[maxScores == -float("Inf")] = 0
    maxScoresExpanded = maxScores.view(vec.shape[0] ,1 , vec.shape[2]).expand(vec.shape[0], vec.shape[1], vec.shape[2])
    return maxScores + torch.log(torch.sum(torch.exp(vec - maxScoresExpanded), 1))


def gen_forget_rate(num_epochs,neg_noise_rate,pos_noise_rate,num_gradual_neg,num_gradual_pos):
    
    forget_rate_neg=neg_noise_rate
    rate_schedule_neg = np.ones(num_epochs) * forget_rate_neg
    rate_schedule_neg[:num_gradual_neg] = np.linspace(0, forget_rate_neg, num_gradual_neg)
    
    forget_rate_pos=pos_noise_rate
    rate_schedule_pos = np.ones(num_epochs) * forget_rate_pos
    rate_schedule_pos[:num_gradual_pos] = np.linspace(0, forget_rate_pos, num_gradual_pos)
    
    return rate_schedule_neg,rate_schedule_pos

def gen_forget_rate_warmup(num_epochs,neg_noise_rate,pos_noise_rate,warm_up_num,num_gradual_neg,num_gradual_pos):
    
    warm_up=[0.0]*warm_up_num

    forget_rate_neg=neg_noise_rate
    rate_schedule_neg = np.ones(num_epochs-warm_up_num) * forget_rate_neg
    rate_schedule_neg[:num_gradual_neg] = np.linspace(0, forget_rate_neg, num_gradual_neg)
    rate_schedule_neg=warm_up+list(rate_schedule_neg)
    
    forget_rate_pos=pos_noise_rate
    rate_schedule_pos = np.ones(num_epochs-warm_up_num) * forget_rate_pos
    rate_schedule_pos[:num_gradual_pos] = np.linspace(0, forget_rate_pos, num_gradual_pos)
    rate_schedule_pos=warm_up+list(rate_schedule_pos)
    
    return rate_schedule_neg,rate_schedule_pos


def lr_decay(config, optimizer: optim.Optimizer, epoch: int) -> optim.Optimizer:
    """
    Method to decay the learning rate
    :param config: configuration
    :param optimizer: optimizer
    :param epoch: epoch number
    :return:
    """
    lr = config.learning_rate / (1 + config.lr_decay * (epoch - 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('learning rate is set to: ', lr)
    return optimizer


# def load_elmo_vec(file: str, insts: List[Instance]):
#     """
#     Load the elmo vectors and the vector will be saved within each instance with a member `elmo_vec`
#     :param file: the vector files for the ELMo vectors
#     :param insts: list of instances
#     :return:
#     """
#     f = open(file, 'rb')
#     for inst in insts:
#         vec=pickle.load(f)[0][0]
#         inst.elmo_vec = vec.detach().numpy()
#         size = vec.shape[1]
#         assert(vec.shape[0] == len(inst.input.words))
#     return size

def get_optimizer(config: Config, model: nn.Module):
    params = model.parameters()
    if config.optimizer.lower() == "sgd":
        print(
            colored("Using SGD: lr is: {}, L2 regularization is: {}".format(config.learning_rate, config.l2), 'yellow'))
        return optim.SGD(params, lr=config.learning_rate, weight_decay=float(config.l2))
    elif config.optimizer.lower() == "adam":
        print(colored("Using Adam", 'yellow'))
        return optim.Adam(params)
    else:
        print("Illegal optimizer: {}".format(config.optimizer))
        exit(1)


def write_results(filename: str, insts):
    f = open(filename, 'w', encoding='utf-8')
    for inst in insts:
        for i in range(len(inst.input)):
            words = inst.input.words
            output = inst.output
            prediction = inst.prediction
            assert len(output) == len(prediction)
            f.write("{}\t{}\t{}\t{}\n".format(i, words[i], output[i], prediction[i]))
        f.write("\n")
    f.close()