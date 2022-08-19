import torch
import random
import numpy as np
from numpy import mean
from typing import Tuple
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
from evaluation.eval import *
from evaluation.postscore import *
from evaluation.score import *
from clmodel.cl import NNCRF_sl


def train_CLout(config: Config, data_collator: Collator, 
                               train_insts: MapStyleJsonDataset, 
                               dev_insts: MapStyleJsonDataset, 
                               test_insts: MapStyleJsonDataset):
    train_num = sum([len(insts) for insts in train_insts])
    print("[Training Info] number of instances: %d" % (train_num))
    # prepare the data 
    dev_batches = batching_list_iterator(config, dev_insts, data_collator)
    test_batches = batching_list_iterator(config, test_insts, data_collator)
    
    #===== set up the model and results saving folder
    model_folder = config.model_folder
    res_folder=config.res_folder
    print("[Training Info] The model will be saved to: %s" % (model_folder))
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)
    
    #===== set up the metrics to save
    #  metrics of length of epoches per outer iteration  # {metric: [[]]}
    inner_metrics = {}     

    #====  specify the cutoff parameter tau scheme
    # get the noisy instance proportion in the training data
    neg_noise_rate_gold, pos_noise_rate_gold = ratio_estimation_iterator(config, train_insts)
    
    # warm start the toss cutoff with ground truth noisy rate
    if(config.neg_noise_rate>=0):
        neg_noise_rate=config.neg_noise_rate
    else:
        neg_noise_rate=neg_noise_rate_gold
    
    if(config.pos_noise_rate>=0):
        pos_noise_rate=config.pos_noise_rate
    else:
        pos_noise_rate=pos_noise_rate_gold
    
     
    print('negative noise rate: '+str(neg_noise_rate))
    print('positve noise rate: '+str(pos_noise_rate))

    if(config.warm_up_num==0) and config.num_gradual_neg<=config.num_epochs:
        rate_schedule_neg, rate_schedule_pos=gen_forget_rate(config.num_epochs,neg_noise_rate,pos_noise_rate,config.num_gradual_neg,config.num_gradual_pos)
    elif (config.warm_up_num+config.num_gradual_neg<=config.num_epochs):
        rate_schedule_neg, rate_schedule_pos=gen_forget_rate_warmup(config.num_epochs,neg_noise_rate,pos_noise_rate,config.warm_up_num,config.num_gradual_neg,config.num_gradual_pos)
    else:
        rate_schedule_neg = np.zeros(config.num_epochs)
        rate_schedule_pos = np.zeros(config.num_epochs)

    if (config.warm_up_num==-1):
        rate_schedule_neg = np.ones(config.num_epochs)*neg_noise_rate
        rate_schedule_pos = np.ones(config.num_epochs)*pos_noise_rate
            
    #==== split the training data
    num_insts_in_fold = math.ceil(len(train_insts) / config.num_folds)
    train_insts_ids = [list(range(i * num_insts_in_fold, (i + 1) * num_insts_in_fold)) for i in range(config.num_folds)]
            
    model_names = [] # model names for each fold
    
    #===== train one iteration for each fold
    train_metrics = [{},{}]
    test_metrics = [{},{}]
    dev_metrics = [{},{}]
    best_dev_f1 = [{},{}]
    models = [None,None]
    raw_ids = train_insts.ids
    for fold_id, folded_train_ids in enumerate(train_insts_ids):
        print(f"[Training Info] Training fold {fold_id}.")
        model_name = model_folder + f"/fold_model_{fold_id}.m"
        model_names.append(model_name)
        #======= batch the train dataset
        train_insts.setids([raw_ids[k] for k in folded_train_ids])
        train_batches = batching_list_iterator(config, train_insts, data_collator)
        train_insts.setids(raw_ids)
        models[fold_id], train_metrics[fold_id], dev_metrics[fold_id], test_metrics[fold_id], best_dev_f1[fold_id] = train_one_CLout(config=config, train_batches = train_batches,
                  dev_insts=dev_insts, dev_batches=dev_batches, 
                  test_insts=test_insts, test_batches=test_batches, 
                  model_name=model_name,
                  rate_schedule_neg=rate_schedule_neg,
                  rate_schedule_pos=rate_schedule_pos)
    
    train_insts.setids(raw_ids)
    
    # get the confidence score performance
    for mname in train_metrics[0].keys(): 
        if mname.startswith('conf'):
            if 'train_'+mname not in inner_metrics:
                inner_metrics['train_'+mname] = []  
            inner_metrics['train_'+mname].append([(a+b)/2 for a,b in zip(train_metrics[0][mname],train_metrics[1][mname])])

    # get the performance before changing anything
    for mname in train_metrics[0].keys(): 
        if 'train_before_'+mname not in inner_metrics:
            inner_metrics['train_before_'+mname] = []  
        inner_metrics['train_before_'+mname].append([(a+b)/2 for a,b in zip(train_metrics[0][mname],train_metrics[1][mname])])

    for mname in test_metrics[0].keys():   
        if 'test_before_'+mname not in inner_metrics:
            inner_metrics['test_before_'+mname] = []
        inner_metrics['test_before_'+mname].append([(a+b)/2 for a,b in zip(test_metrics[0][mname],test_metrics[1][mname])])

    for mname in dev_metrics[0].keys():   
        if 'dev_before_'+mname not in inner_metrics:
            inner_metrics['dev_before_'+mname] = []
        inner_metrics['dev_before_'+mname].append([(a+b)/2 for a,b in zip(dev_metrics[0][mname],dev_metrics[1][mname])])

    print(best_dev_f1)

    #====== modify the training data accordingly
    if config.modify == 'correct':
         #==== assign hard prediction to other folds
        print("\n\n[Data Info] Assigning labels for the HARD approach")

        for fold_id, folded_train_ids in enumerate(train_insts_ids):
            model = models[fold_id]
            train_insts.setids([raw_ids[k] for k in folded_train_ids])
            train_batches = batching_list_iterator(config, train_insts, data_collator)
            # change the label of the other trainning fold 
            train_insts.setids(raw_ids)
            hard_constraint_predict(config=config, model=model,
                                    fold_batches = train_batches,
                                    folded_insts = train_insts,
                                    folded_ids= folded_train_ids)  ## set a new label id
        train_insts.ids = raw_ids

    if config.modify == 'rank':
        scores = []
        for fold_id, folded_train_ids in enumerate(train_insts_ids):
            model = models[fold_id]
            # rank samples of the other trainning fold 
            train_insts.setids([raw_ids[k] for k in folded_train_ids])
            train_batches = batching_list_iterator(config, train_insts, data_collator)
            train_insts.setids(raw_ids)
            scores.extend(ranknoisy(config=config, model=model, 
                                    fold_batches = train_batches, 
                                    folded_insts = train_insts,
                                    folded_ids= folded_train_ids)) ## set a new label id
        sorted_ids = np.argsort(scores) # we put the one with higher scores ahead (more confident)
        train_insts.setids([raw_ids[k] for k in sorted_ids])

    if config.modify == 'weight':
        # to be tested and do corresponding loss change
        for fold_id, folded_train_ids in enumerate(train_insts_ids):
            model = models[fold_id]
            train_insts.setids([raw_ids[k] for k in folded_train_ids])
            train_batches = batching_list_iterator(config, train_insts, data_collator)
            train_insts.setids(raw_ids)
            # weight samples of the other trainning fold 
            weightnoisy(config=config, model=model, 
                        fold_batches = train_batches, 
                        folded_insts = train_insts,
                        folded_ids= folded_train_ids)  ## set a new label id
        train_insts.setids(raw_ids)

    print("\n\n")

    #==== train the final model

    print("[Training Info] Training the final model" )
    
    # # save the trainset as well
    f = open(res_folder+'/train_modified.pkl', 'wb')
    pickle.dump(train_insts, f)
    f.close() 
    
    
    neg_noise_rate_after, pos_noise_rate_after = ratio_estimation_iterator(config, train_insts)
    print('negative noise rate: '+str(neg_noise_rate_after))
    print('positve noise rate: '+str(pos_noise_rate_after))
    
    # batching again
    all_train_batches = batching_list_iterator(config, train_insts, data_collator)
    
    # to save the model with best dev f1 so far
    model_name = model_folder + "/"+"best_model.m"
    config_name = model_folder + "/"+"best_config.conf"

    model, train_metrics, dev_metrics, test_metrics, best_dev_f1_final = train_one_CLout(config = config, 
                      train_batches=all_train_batches, 
                      dev_insts=dev_insts, 
                      dev_batches=dev_batches,
                      model_name=model_name, 
                      config_name=config_name,
                      test_insts=test_insts, 
                      test_batches=test_batches,
                      rate_schedule_neg=None, # don't throw out any sample in the final training
                      rate_schedule_pos=None,
                      train_insts=train_insts)
    
    for mname in train_metrics.keys(): 
        if 'train_'+mname not in inner_metrics:
            inner_metrics['train_'+mname] = []  
        inner_metrics['train_'+mname].append(train_metrics[mname])

    for mname in test_metrics.keys():   
        if 'test_'+mname not in inner_metrics:
            inner_metrics['test_'+mname] = []
        inner_metrics['test_'+mname].append(test_metrics[mname])

    for mname in dev_metrics.keys():   
        if 'dev_'+mname not in inner_metrics:
            inner_metrics['dev_'+mname] = []
        inner_metrics['dev_'+mname].append(dev_metrics[mname])

    inner_metrics['neg_noise_rate_gold'] = neg_noise_rate_gold
    inner_metrics['pos_noise_rate_gold'] = pos_noise_rate_gold   
    inner_metrics['neg_noise_rate_after'] = neg_noise_rate_after
    inner_metrics['pos_noise_rate_after'] = pos_noise_rate_after 

    return inner_metrics


def train_one_CLout(config: Config, 
              train_batches: List[dict], 
              dev_insts: MapStyleJsonDataset,
              dev_batches: List[dict], 
              model_name: str, 
              test_insts: MapStyleJsonDataset = None,
              test_batches: MapStyleJsonDataset = None, 
              config_name: str = None, 
              rate_schedule_neg=None,
              rate_schedule_pos=None,
              train_insts = None):
   
    epoch = config.num_epochs
    best_dev_f1 = -1 # save the best one
    saved_test_metrics = None  
    
    # set up metrics to save
    allmetrics = []
    for metric in ['precision','recall','f1']:
        allmetrics.extend([kind+metric for kind in ['','hard_','easy_','high_','low_','head_','tail_']])

    if config.diagonosis and train_insts:
        train_allmetrics = allmetrics.copy()
        for metric in ['precision','recall','f1']:
            train_allmetrics.extend([kind+metric for kind in ['noise_','clean_']])
        train_metrics_list = {metric:[] for metric in ['loss','neg_loss','pos_loss']+train_allmetrics}  
    else:
        train_metrics_list = {metric:[] for metric in ['loss','neg_loss','pos_loss']}
    
    test_metrics_list = {metric:[] for metric in allmetrics}
    dev_metrics_list = {metric:[] for metric in allmetrics}

    #==== training
    model = NNCRF_sl(config) # initilize a new model
    model.train()
    
    optimizer = get_optimizer(config, model)
    
    # get optimizer and lr rate scheduler
    if config.encoder =='bilstm':
        optimizer = get_optimizer(config, model)
    else:    
        optimizer, scheduler = get_huggingface_optimizer_and_scheduler(config, model, num_training_steps=len(train_batches) * epoch,  
                                        weight_decay=0.0, eps = 1e-8, warmup_step=0) 
    
    for i in range(1, epoch + 1):
        #  to save the sliced (entity & non-entity) performance of is_confidence prediction
        #  {neg_recall,pos_recall,neg_precision,pos_precision,neg_f1,pos_f1}
        if config.cutoff=='oracle':
            conf_metrics = {metric: 0 for metric in ['pos_entity','pos_predict','neg_entity','neg_predict',
                                                    'pos_p','neg_p','pos_fit_predict','pos_fit_p',
                                                    'neg_fit_predict','neg_fit_p']}
        else:
            conf_metrics = {metric: 0 for metric in ['pos_entity','pos_predict','neg_entity','neg_predict',
                                                    'pos_p','neg_p']}       
        
        # the toss rate for nonentity and entity for each epoch
        if rate_schedule_neg is not None:
            forget_rate_neg = rate_schedule_neg[i-1]
        else:
            forget_rate_neg = 0    
        if rate_schedule_pos is not None:
            forget_rate_pos = rate_schedule_pos[i-1]
        else:
            forget_rate_pos = 0    

                
        epoch_loss = 0
        epoch_loss_neg = 0
        epoch_loss_pos = 0

        start_time = time.time()
        model.zero_grad()

        is_constrain=config.is_constrain
        
        # train and compute loss and confidence metrics
        if config.modify == 'rank':
            indexlist = range(len(train_batches))
        else:
            indexlist = np.random.permutation(len(train_batches)) 
                
        for index in indexlist:
            model.train()
            
            # input data and parameters to the forward function
            tmp=tuple([train_batches[index], forget_rate_neg,forget_rate_pos,is_constrain,True]) # the last is whether_train
            
            # get the metrics: 
            loss,confmetrics,loss_neg,loss_pos,confscore= model(*tmp)
            
            conf_metrics = { metric: conf_metrics[metric]+confmetrics[metric] for metric in confmetrics.keys()}
            epoch_loss += loss.item()
            epoch_loss_neg += loss_neg.item()
            epoch_loss_pos += loss_pos.item()

            if config.diagonosis or config.cutoff in ['goracle','fitmix']: # attach the computed confidence score on the fly
                if model.score in ['nerloss','encoderloss','diff']:
                    #one_batch_insts = train_insts[index * config.batch_size: (index + 1) * config.batch_size]
                    word_seq_lens = train_batches[index]['word_seq_lens'].cpu().numpy()
                    for idx in range(len(word_seq_lens)):
                        inst_idx = index * config.batch_size+idx
                        length = word_seq_lens[idx]
                        train_insts.update(inst_idx, 'scores', confscore[idx,:length])
            
            loss.backward()
            if config.encoder=='bilstm':
                optimizer.step()
            else:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()    
            model.zero_grad()


        confscore_metrics = {}
        for tt in ['pos','neg']:
            confscore_metrics[tt+'_conf_precision'] = conf_metrics[tt+'_p'] * 1.0 / conf_metrics[tt+'_predict'] * 100 if  conf_metrics[tt+'_predict'] != 0 else 0
            confscore_metrics[tt+'_conf_recall'] = conf_metrics[tt+'_p'] * 1.0 / conf_metrics[tt+'_entity'] * 100 if  conf_metrics[tt+'_entity'] != 0 else 0
            confscore_metrics[tt+'_conf_f1'] =  2.0 * confscore_metrics[tt+'_conf_precision'] * confscore_metrics[tt+'_conf_recall'] / (confscore_metrics[tt+'_conf_precision'] + confscore_metrics[tt+'_conf_recall']) if confscore_metrics[tt+'_conf_precision'] != 0 or confscore_metrics[tt+'_conf_recall'] != 0 else 0
        confscore_metrics['precision'] = (confscore_metrics['neg_conf_precision']+confscore_metrics['pos_conf_recall'])/2
        confscore_metrics['recall'] = (confscore_metrics['neg_conf_recall']+confscore_metrics['pos_conf_recall'])/2
        confscore_metrics['f1'] = (confscore_metrics['neg_conf_f1']+confscore_metrics['pos_conf_f1'])/2
        
        if config.cutoff=='oracle':
            for tt in ['pos','neg']:
                confscore_metrics[tt+'_fit_precision'] = conf_metrics[tt+'_fit_p'] * 1.0 / conf_metrics[tt+'_fit_predict'] * 100 if  conf_metrics[tt+'_fit_predict'] != 0 else 0
                confscore_metrics[tt+'_fit_recall'] = conf_metrics[tt+'_fit_p'] * 1.0 / conf_metrics[tt+'_entity'] * 100 if  conf_metrics[tt+'_entity'] != 0 else 0
                confscore_metrics[tt+'_fit_f1'] =  2.0 * confscore_metrics[tt+'_fit_precision'] * confscore_metrics[tt+'_fit_recall'] / (confscore_metrics[tt+'_fit_precision'] + confscore_metrics[tt+'_fit_recall']) if confscore_metrics[tt+'_fit_precision'] != 0 or confscore_metrics[tt+'_fit_recall'] != 0 else 0
            confscore_metrics['fit_precision'] = (confscore_metrics['neg_fit_precision']+confscore_metrics['pos_fit_recall'])/2
            confscore_metrics['fit_recall'] = (confscore_metrics['neg_fit_recall']+confscore_metrics['pos_fit_recall'])/2
            confscore_metrics['fit_f1'] = (confscore_metrics['neg_fit_f1']+confscore_metrics['pos_fit_f1'])/2

        epoch_loss /= len(train_batches)
        epoch_loss_neg /= len(train_batches)
        epoch_loss_pos /= len(train_batches)

        end_time = time.time()
        
        
        if config.cutoff=='oracle': 
            print("Epoch %d: loss %.5f, conf %.1f, fit %.1f Time is %.2fs" % (i, epoch_loss, confscore_metrics['f1'], confscore_metrics['fit_f1'], end_time - start_time), flush=True)
        else:
            print("Epoch %d: loss %.5f, conf %.1f, Time is %.1fs" % (i, epoch_loss, confscore_metrics['f1'],  end_time - start_time), flush=True)

        train_metrics_list['loss'].append(epoch_loss)
        train_metrics_list['neg_loss'].append(epoch_loss_neg)
        train_metrics_list['pos_loss'].append(epoch_loss_pos)
        
        # #========only add the conf_metric to results when we toss some sample
        if (rate_schedule_neg is not None) or (rate_schedule_pos is not None):
            for metric in confscore_metrics.keys(): 
                if 'conf_'+metric not in train_metrics_list:
                    train_metrics_list['conf_'+metric] = []
                train_metrics_list['conf_'+metric].append(confscore_metrics[metric]) 
 
        #======== evaluate the model using dev and test data
        model.eval()

        # evalaute the NER task
        if config.diagonosis and train_insts:
            train_metrics = evaluate_model(config, model, train_batches, "train", train_insts)
            
        dev_metrics = evaluate_model(config, model, dev_batches, "dev", dev_insts)
        
        if test_insts is not None:
            test_metrics = evaluate_model(config, model, test_batches, "test", test_insts)
        else:
            test_metrics = None    
        
        if model_name:
            if dev_metrics['f1'] > best_dev_f1:
                print("saving the best model so far for all iteration...")
                best_dev_f1 = dev_metrics['f1']
                if test_insts is not None:
                    saved_test_metrics = test_metrics
                    with open(config.res_folder+'/bestresults.txt', 'w') as f:
                        for key, item in test_metrics.items():
                            f.write("%s %.5f \n" % (key, item))
                        f.close()
                #torch.save(model.state_dict(), model_name)
                # # Save the corresponding config as well.
                if config_name:
                    f = open(config_name, 'wb')
                    pickle.dump(config, f)
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


    if test_insts is not None:
        print(f"The best dev F1: {best_dev_f1}")
        print(f"The corresponding test: {saved_test_metrics}")    
    
    return model, train_metrics_list, dev_metrics_list, test_metrics_list, best_dev_f1

