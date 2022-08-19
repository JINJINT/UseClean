from datastruct.dataset import Collator, MapStyleJsonDataset, batching_list_iterator
import torch
import random
import numpy as np
from numpy import mean
from typing import Tuple, Dict, List
import time
from typing import List
from termcolor import colored
import os
import pickle
import tarfile
import shutil
import math
import itertools
import matplotlib.pyplot as plt
from itertools import chain


# from datastruct import Instance
from config import Config
# from datastruct import Reader, ContextEmb
from utils.utils import *
from utils.trans_utils import *
from evaluation.eval import *
from clmodel.coreg import NNCRF_coreg


def train_coreg(config: Config, data_collator: Collator, 
                                   train_insts: MapStyleJsonDataset, 
                                   dev_insts: MapStyleJsonDataset, 
                                   test_insts: MapStyleJsonDataset):
    """
    train the baseline without confident learning
    """
        
    train_num = len(train_insts)
    print("[Training Info] number of instances: %d" % (train_num))

    #===== set up the model and results saving folder
    model_folder = config.model_folder
    res_folder=config.res_folder
    print("[Training Info] The model will be saved to: %s.tar.gz" % (model_folder))
    
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)    

    #===== set up the metrics to save
    metrics = {}
    
    #===== batch and tensorize the dataset
    
    dev_batches = batching_list_iterator(config, dev_insts, data_collator)
    test_batches = batching_list_iterator(config, test_insts, data_collator)
    train_batches = batching_list_iterator(config, train_insts, data_collator)
    
    #==== to save the model with best dev f1 so far
    model_name = model_folder + "/"+"best_lstm_crf.m"
    config_name = model_folder + "/"+"best_config.conf"
    res_name = res_folder + "/"+"best_lstm_crf.results".format()

    #==== train the model
    #if not os.path.exists(model_name):
 
    train_metrics, dev_metrics, test_metrics, best_dev_f1 = train_one_coreg(config = config, 
                            train_batches=train_batches, 
                            dev_insts=dev_insts, 
                            dev_batches=dev_batches,
                            model_name=model_name,
                            config_name=config_name,
                            test_insts=test_insts, 
                            test_batches=test_batches,
                            bestf1 = -1,
                            train_insts=train_insts)


    for mname in train_metrics.keys():   
        metrics['train_'+mname] = train_metrics[mname]

    for mname in test_metrics.keys():   
        metrics['test_'+mname] = test_metrics[mname]

    for mname in dev_metrics.keys():   
        metrics['dev_'+mname] = dev_metrics[mname]
    

    print(f'Best dev f1: {best_dev_f1}')

    return metrics

def train_one_coreg(config: Config, 
              train_batches: List[dict], 
              dev_insts: MapStyleJsonDataset,
              dev_batches: List[dict], 
              model_name: str, 
              config_name: str = None, 
              test_insts: MapStyleJsonDataset = None,
              test_batches: List[dict] = None, 
              bestf1 = -1,
              train_insts = None):  

    epoch = config.num_epochs
    best_dev_f1 = bestf1 # save the best one
    saved_test_metrics = None  

    # parepare to save metrics
    allmetrics = ['loss']
    for metric in ['precision','recall','f1']:
        allmetrics.extend([kind+metric for kind in ['','hard_','easy_','high_','low_','head_','tail_']])
    
    if config.diagonosis and train_insts:
        train_allmetrics = allmetrics.copy()
        for metric in ['precision','recall','f1']:
            train_allmetrics.extend([kind+metric for kind in ['noise_','clean_']])
        train_metrics_list = {metric:[] for metric in train_allmetrics}  
    else:
        train_metrics_list = {metric:[] for metric in ['loss']}
    
    test_metrics_list = {metric:[] for metric in allmetrics}
    dev_metrics_list = {metric:[] for metric in allmetrics} 
    
    #==== training
    # initilize a new model
    model = NNCRF_coreg(config) 
    model.train()
    
    # get optimizer and lr rate scheduler
    if config.encoder =='bilstm':
        optimizer = get_optimizer(config, model)
    else:    
        optimizer, scheduler = get_huggingface_optimizer_and_scheduler(config, model, num_training_steps=len(train_batches) * epoch,
                                                                   weight_decay=0.0, eps = 1e-8, warmup_step=0)
    # training
    for i in range(1, epoch + 1):          
        epoch_loss = 0
        testepoch_loss = 0
        devepoch_loss = 0
        
        start_time = time.time()
        model.zero_grad()

        if config.encoder=='bilstm':
            optimizer = lr_decay(config, optimizer, i)
        
        for index in np.random.permutation(len(train_batches)):
            model.train()

            if i < config.alpha_warmup_ratio * config.num_epochs:
                alpha_t = 0.0
            else:
                alpha_t = config.alpha * (1.1**i)
            
            # input data and parameters to the forward function            
            # get the metrics: 
            loss= model(train_batches[index], alpha_t)
            epoch_loss += loss.item()
            loss.backward()
            # if config.max_grad_norm > 0:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            if config.encoder=='bilstm':
                optimizer.step()
            else:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            model.zero_grad()

        epoch_loss /= len(train_batches)

        end_time = time.time()
        print("Epoch %d: loss %.5f, Time is %.2fs" % (i, epoch_loss, end_time - start_time), flush=True)
    
        train_metrics_list['loss'].append(epoch_loss)
        
        #======== evaluate the model using dev and test data
        model.eval()

        for index in np.random.permutation(len(test_batches)):
            
            # input data and parameters to the forward function
            # get the metrics: 
            testloss= model(test_batches[index])
            testepoch_loss += testloss.item()   
        
        testepoch_loss /= len(test_batches)

        test_metrics_list['loss'].append(testepoch_loss)

        for index in np.random.permutation(len(dev_batches)):
            
            devloss = model(dev_batches[index])
            devepoch_loss += devloss.item()  
        
        devepoch_loss /= len(dev_batches)

        dev_metrics_list['loss'].append(devepoch_loss)

        #====== get other metrics
        # metric is [precision, recall, f_score]
        if config.diagonosis and train_insts is not None:
            train_metrics,_,_,_ = evaluate_model(config, model, train_batches, "train", train_insts)
            print('noise f1: %.5f'%(train_metrics['noise_f1']))
        dev_metrics = evaluate_model(config, model, dev_batches, "dev", dev_insts)
        test_metrics = evaluate_model(config, model, test_batches, "test", test_insts)
        
        if dev_metrics['f1'] > best_dev_f1:
            print("saving the best model so far for all epoches...")
            best_dev_f1 = dev_metrics['f1']
            if config.encoder=='bilstm':
                torch.save(model.state_dict(), model_name)  
            if config_name:
                f = open(config_name, 'wb')
                pickle.dump(config, f)
                f.close()
            if test_insts is not None:
                saved_test_metrics = test_metrics
                with open(config.res_folder+'/bestresults.txt', 'w') as f:
                    for key, item in test_metrics.items():
                        f.write("%s %.5f \n" % (key, item))
                    f.close()  
            if config.encoder=='bilstm':
                torch.save(model.state_dict(), model_name)  

        model.zero_grad()

        for metric in dev_metrics:
            dev_metrics_list[metric].append(dev_metrics[metric])
        
        if test_insts is not None:
            for metric in test_metrics:
                test_metrics_list[metric].append(test_metrics[metric])

        if config.diagonosis and train_insts is not None:
            for metric in train_metrics:
                train_metrics_list[metric].append(train_metrics[metric])        
        
    
    print(f"The best dev F1: {best_dev_f1}")
    print(f"The corresponding test: {saved_test_metrics}")    
    
    return train_metrics_list, dev_metrics_list, test_metrics_list, best_dev_f1
