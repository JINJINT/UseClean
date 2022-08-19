from datastruct import dataset
import torch
import random
import numpy as np
from numpy import mean
from typing import Tuple, List
import time
from typing import List
import os
import pickle
import tarfile
import shutil
import math
import itertools
from itertools import chain

# from datastruct import Instance
from datastruct.dataset import *
from config import Config
# from datastruct import Reader, ContextEmb
from utils.utils import *
from utils.trans_utils import *
from evaluation.eval import *
from evaluation.postscore import *
from clmodel.cl import NNCRF_sl
from clmodel.baseline import NNCRF_baseline
from clmodel.coreg import NNCRF_coreg
import copy
from main import plotinnermetrics
import seaborn as sns
from processer.tokenization import _read_txt, build_emb_table, build_word_idx, read_pretrain_embedding


def assign_cross_score(config: Config, train_insts: MapStyleJsonDataset, 
                                       dev_insts: MapStyleJsonDataset, 
                                       data_collator: Collator):
    
    train_num = len(train_insts)
    print("[Training Info] number of instances: %d" % (train_num))
    num_folds = config.crossfolds # used as default in original paper
    num_dev = len(dev_insts)//num_folds

    #dev = [dev_insts[i] for i in random.sample(range(len(dev_insts)),num_dev)]
    dev_b = batching_list_iterator(config, dev_insts, data_collator)
    #===== set up the model and results saving folder
    model_folder = config.model_folder

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    #===== train the model for each folds
    info = create_folds(config, train_insts, num_folds)
    models = []
    out_batches_list = []
    all_train_ids = train_insts.ids.copy()
    for i in range(num_folds):
        in_indexs = info[f'fold-{i}']['train_indexs']
        out_indexs = info[f'fold-{i}']['test_indexs']
        train_insts.setids([all_train_ids[_in] for _in in in_indexs])
        in_batches = batching_list_iterator(config, train_insts, data_collator)
        out_batches_list.append(out_indexs)        
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
 
        if(config.warm_up_num==0):
            rate_schedule_neg, rate_schedule_pos=gen_forget_rate(config.num_epochs,neg_noise_rate,pos_noise_rate,config.num_gradual_neg,config.num_gradual_pos)
        else:
            rate_schedule_neg, rate_schedule_pos=gen_forget_rate_warmup(config.num_epochs,neg_noise_rate,pos_noise_rate,config.warm_up_num,config.num_gradual_neg,config.num_gradual_pos)

        model = train_one_cross(config = config, 
                      train_batches = in_batches, 
                      dev_batches = dev_b,
                      dev_insts = dev_insts,
                      rate_schedule_neg=rate_schedule_neg,
                      rate_schedule_pos=rate_schedule_pos)
        models.append(model)              

    #===== get the confidence score as out-of-sample prediction error
    print("\n\n[Data Info] Getting out-of-sample confidence score")
    train_insts.setids(all_train_ids)
    for fold_id in range(num_folds):
        model = models[fold_id]
        out_indexs = info[f'fold-{fold_id}']['test_indexs']
        train_insts.setids([all_train_ids[_in] for _in in out_indexs])
        out_batches = batching_list_iterator(config, train_insts, data_collator)
        train_insts.setids(all_train_ids)
        # attach the score 
        attach_cross_score(config=config, model=model,
                        fold_batches = out_batches,
                        folded_insts = train_insts,
                        folded_ids = out_indexs)
                

def train_one_cross(config: Config, 
              train_batches: List[dict], 
              dev_batches: List[dict],
              dev_insts: MapStyleJsonDataset, 
              epoch = 5,
              rate_schedule_neg=None,
              rate_schedule_pos=None):  

    model = NNCRF_sl(config) # initilize a new model
    model.score = 'nerloss'
    # model = torch.nn.DataParallel(model) # !!! distributed 
    # model.to(config.device)
    
    model.train()
   # get optimizer and lr rate scheduler
    if config.encoder =='bilstm':
        optimizer = get_optimizer(config, model)
    else:    
        optimizer, scheduler = get_huggingface_optimizer_and_scheduler(config, model, num_training_steps=len(train_batches) * epoch,
                                                                   weight_decay=0.0, eps = 1e-8, warmup_step=0)

    epoch = config.num_epochs
    best_dev_f1 = -1 # save the best one
    
    for i in range(1, epoch + 1):
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
        
        start_time = time.time()
        model.zero_grad()
        
        if config.encoder=='bilstm':
            optimizer = lr_decay(config, optimizer, i)
        
        for index in np.random.permutation(len(train_batches)):
            model.train()
            
            # input data and parameters to the forward function
            tmp=tuple([train_batches[index],forget_rate_neg,forget_rate_pos,config.is_constrain,True])
            
            # get the metrics: 
            loss,_,_,_,_= model(*tmp)
            
            epoch_loss += loss.item()
            
            loss.backward()
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
        
        #======== evaluate the model using dev and test data
        model.eval()
        dev_metrics = evaluate_model(config, model, dev_batches, "dev", dev_insts)
        
        if dev_metrics['f1'] >= best_dev_f1:
            print("saving the best model so far for all epoches...")
            best_dev_f1 = dev_metrics['f1']
            best_model = copy.deepcopy(model)
            #torch.save(model.state_dict(), model_name)   

        model.zero_grad()
    
    print(f"The best dev F1: {best_dev_f1}") 
    return best_model
    

def assign_useclean_score(config: Config, 
                          train_insts: MapStyleJsonDataset, 
                          dev_insts: MapStyleJsonDataset, 
                          test_insts: MapStyleJsonDataset, 
                          data_collator: Collator,
                          sampletype = 'random'):
    
    cleanprop = config.cleanprop
    num_clean = max(10, int(len(train_insts)*cleanprop))
    
    if sampletype == 'random':
        # random sample
        if config.dataset!='wikigold_self':
            index_clean = random.sample(range(len(train_insts)),num_clean)
        else:
            index_clean = random.sample(range(600),num_clean)    
        
    if sampletype == 'tail':  
        if config.dataset.startswith('conll') or config.dataset.startswith('wiki'):
            tailentity = set(['MISC','LOC'])

        if config.dataset.startswith('massive'):      
            tailentity = set(['relation', 'timeofday', 'house_place', 'music_genre', 'business_type', 'player_setting', 'audiobook_name', 'game_name', 
                                            'podcast_descriptor', 'email_address', 'general_frequency', 'playlist_name', 'podcast_name', 'order_type', 'personal_info', 
                                            'color_type', 'change_amount', 'time_zone', 'music_descriptor', 'meal_type', 'app_name', 'joke_type', 'transport_agency', 'movie_name', 
                                            'coffee_type', 'ingredient', 'email_folder', 'transport_name', 'alarm_type', 'cooking_type', 'movie_type', 'audiobook_author', 'transport_descriptor', 'drink_type', 'sport_type', 'music_album', 'game_type'])
        def checktail(gout):
            for g in gout:
                if g[2:] in tailentity:
                    return True
            return False    
        allgolden = [i for i in range(len(train_insts)) if checktail(train_insts[i]['gold_labels'])]
        index_clean = random.sample(allgolden, num_clean)
    
    if sampletype == 'head':   
        if config.dataset.startswith('conll') or config.dataset.startswith('wiki'):
            headentity = set(['PER','ORG']) 
        
        if config.dataset.startswith('massive'):          
            headentity = set(['date', 'time'])

        def checkhead(gout):
            for g in gout:
                if g[2:] in headentity:
                    return True
            return False    
        allgolden = [i for i in range(len(train_insts)) if checkhead(train_insts[i]['gold_labels'])]
        index_clean = random.sample(allgolden, num_clean)  


    #clean = [train_insts[i] for i in  index_clean] 
    index_rest = [i for i in range(len(train_insts)) if i not in index_clean]
    
    for i in index_clean:
        train_insts.update(i,'labels_id', train_insts[i]['gold_labels_id'])
        train_insts.update(i, 'labels', train_insts[i]['gold_labels'])
    
    #newtrain_insts = [train_insts[i] for i in range(len(train_insts)) if i not in index_clean]    
    all_train_ids = train_insts.ids.copy()
    train_insts.setids([all_train_ids[_in] for _in in index_clean])
    clean_b = batching_list_iterator(config, train_insts, data_collator)
    dev_b = batching_list_iterator(config, dev_insts, data_collator)
    test_b = batching_list_iterator(config, test_insts, data_collator)
    train_insts.setids([all_train_ids[_in] for _in in index_rest])
    train_b =  batching_list_iterator(config, train_insts, data_collator)

    #===== set up the model and results saving folder
    model_folder = config.model_folder
    model_name = model_folder + '/model_clean.m'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    #===== train the model for each folds
    metrics = {}
    model, dev_metrics, test_metrics, train_metrics = train_one_clean(config = config, 
                    train_batches=clean_b, 
                    dev_batches = dev_b,
                    dev_insts = dev_insts,
                    test_batches = test_b,
                    test_insts = test_insts)
    if config.warm:
        torch.save(model.state_dict(), model_name)          
    
    for mname in dev_metrics.keys():   
        metrics['dev_'+mname] = dev_metrics[mname]

    for mname in test_metrics.keys():   
        metrics['test_'+mname] = test_metrics[mname]  

    for mname in train_metrics.keys():   
        metrics['train_'+mname] = train_metrics[mname]      

    plotinnermetrics(metricslist = ['precision','recall','f1'], inner_metrics = metrics, info='before_total', conf = config)
    plotinnermetrics(metricslist = ['hard_precision','hard_recall','hard_f1'], inner_metrics = metrics,  info='before_hard', conf = config)
    plotinnermetrics(metricslist = ['easy_precision','easy_recall','easy_f1'], inner_metrics = metrics,  info='before_easy', conf = config)
    plotinnermetrics(metricslist = ['head_precision','head_recall','head_f1'], inner_metrics = metrics,  info='before_head', conf = config)
    plotinnermetrics(metricslist = ['tail_precision','tail_recall','tail_f1'], inner_metrics = metrics,  info='before_tail', conf = config)
    plotinnermetrics(metricslist = ['high_precision','high_recall','high_f1'], inner_metrics = metrics, info='before_highfreq', conf = config)
    plotinnermetrics(metricslist = ['low_precision','low_recall','low_f1'], inner_metrics = metrics,  info='before_lowfreq', conf = config)
    plotinnermetrics(metricslist = ['train_loss'], inner_metrics = metrics, info='before_loss', conf = config)

    f = open(config.res_folder+'/cleanmetrics.pkl', 'wb')
    pickle.dump(metrics, f)
    f.close() 


    #===== get the confidence score as out-of-sample prediction error
    print("\n\n[Data Info] Getting confidence score predicted using clean samples")

    train_insts.setids(all_train_ids)
    # change the label of the other trainning fold 
    attach_useclean_score(config=config, model=model,
                    fold_batches = train_b,
                    folded_insts = train_insts,
                    folded_ids = index_rest)                
    train_insts.setids([all_train_ids[_in] for _in in index_rest]) 
    if config.num_epochs==0:
        evaluate_cleanmodel(config, train_b, train_insts)         

    print('length of rest train: %d'%(len(train_insts)))           
    

def train_one_clean(config: Config, 
              train_batches: List[dict], 
              dev_batches: List[dict],
              dev_insts: MapStyleJsonDataset,
              test_batches: List[dict],
              test_insts: MapStyleJsonDataset,
              epoch = 30):  
    model = NNCRF_baseline(config) # initilize a new model
    # model = torch.nn.DataParallel(model) # !!! distributed 
    # model.to(config.device)
    epoch = config.cleanepochs

    model.train()
    if config.encoder =='bilstm':
        optimizer = get_optimizer(config, model)
    else:    
        optimizer, scheduler = get_huggingface_optimizer_and_scheduler(config, model, num_training_steps=len(train_batches) * epoch,
                                                                   weight_decay=0.0, eps = 1e-8, warmup_step=0)

    #epoch = config.num_epochs
    best_dev_f1 = -1 # save the best one
    allmetrics = ['loss']
    for metric in ['precision','recall','f1']:
        allmetrics.extend([kind+metric for kind in ['','hard_','easy_','high_','low_','head_','tail_']])
    
    train_metrics_list = {metric:[] for metric in ['loss']}
    test_metrics_list = {metric:[] for metric in allmetrics}
    dev_metrics_list = {metric:[] for metric in allmetrics} 

    for i in range(1, epoch + 1):
                
        epoch_loss = 0
        
        start_time = time.time()
        model.zero_grad()
        
        if config.encoder=='bilstm':
            optimizer = lr_decay(config, optimizer, i)
        
        for index in np.random.permutation(len(train_batches)):
            model.train()
            
            # input data and parameters to the forward function
            tmp = train_batches[index]
            
            # get the metrics: 
            loss= model(tmp)
            
            epoch_loss += loss.item()
            
            loss.backward()
            if config.encoder=='bilstm':
                optimizer.step()
            else:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            model.zero_grad()

        epoch_loss /= len(train_batches)
        
        train_metrics_list['loss'].append(epoch_loss)

        end_time = time.time()
        print("Epoch %d: loss %.5f, Time is %.2fs" % (i, epoch_loss, end_time - start_time), flush=True)
        
        #======== evaluate the model using dev and test data
        model.eval()
        dev_metrics = evaluate_model(config, model, dev_batches, "dev", dev_insts)
        test_metrics = evaluate_model(config, model, test_batches, "test", test_insts)
        
        if dev_metrics['f1'] >= best_dev_f1:
            print("saving the best model so far for all epoches...")
            best_dev_f1 = dev_metrics['f1']
            best_model = copy.deepcopy(model)
            if test_insts is not None:
                with open(config.res_folder+'/cleanbestresults.txt', 'w') as f:
                    for key, item in test_metrics.items():
                        f.write("%s %.5f \n" % (key, item))
                    f.close()
            #torch.save(model.state_dict(), model_name)   

        model.zero_grad()
    
        for metric in dev_metrics:
            dev_metrics_list[metric].append(dev_metrics[metric])
        
        for metric in test_metrics:
            test_metrics_list[metric].append(test_metrics[metric])     
    
    print(f"The best dev F1: {best_dev_f1}") 

    return best_model, dev_metrics_list, test_metrics_list, train_metrics_list


def assign_usecontrast_score(config: Config, 
                          train_insts: MapStyleJsonDataset, 
                          dev_insts: MapStyleJsonDataset, 
                          test_insts: MapStyleJsonDataset, 
                          data_collator: Collator,
                          sampletype = 'random'):
    
    cleanprop = config.cleanprop
    num_clean = max(10, int(len(train_insts)*cleanprop))
    
    if sampletype == 'random':
        # random sample
        index_clean = random.sample(range(len(train_insts)),num_clean)
        
    if sampletype == 'tail':  
        if config.dataset.startswith('conll'):
            tailentity = set(['MISC','LOC'])

        if config.dataset.startswith('massive'):      
            tailentity = set(['relation', 'timeofday', 'house_place', 'music_genre', 'business_type', 'player_setting', 'audiobook_name', 'game_name', 
                                            'podcast_descriptor', 'email_address', 'general_frequency', 'playlist_name', 'podcast_name', 'order_type', 'personal_info', 
                                            'color_type', 'change_amount', 'time_zone', 'music_descriptor', 'meal_type', 'app_name', 'joke_type', 'transport_agency', 'movie_name', 
                                            'coffee_type', 'ingredient', 'email_folder', 'transport_name', 'alarm_type', 'cooking_type', 'movie_type', 'audiobook_author', 'transport_descriptor', 'drink_type', 'sport_type', 'music_album', 'game_type'])
        def checktail(gout):
            for g in gout:
                if g[2:] in tailentity:
                    return True
            return False    
        allgolden = [i for i in range(len(train_insts)) if checktail(train_insts[i]['gold_labels'])]
        index_clean = random.sample(allgolden, num_clean)
    
    if sampletype == 'head':   
        if config.dataset.startswith('conll'):
            headentity = set(['PER','ORG']) 
        
        if config.dataset.startswith('massive'):          
            headentity = set(['date', 'time'])
        def checkhead(gout):
            for g in gout:
                if g[2:] in headentity:
                    return True
            return False    
        allgolden = [i for i in range(len(train_insts)) if checkhead(train_insts[i]['gold_labels'])]
        index_clean = random.sample(allgolden, num_clean)


    #clean = [train_insts[i] for i in  index_clean] 
    index_rest = [i for i in range(len(train_insts)) if i not in index_clean]
    
    #newtrain_insts = [train_insts[i] for i in range(len(train_insts)) if i not in index_clean]    
    all_train_ids = train_insts.ids.copy()
    train_insts.setids([all_train_ids[_in] for _in in index_clean])
    clean_b = batching_list_iterator(config, train_insts, data_collator)
    dev_b = batching_list_iterator(config, dev_insts, data_collator)
    test_b = batching_list_iterator(config, test_insts, data_collator)

    #===== set up the model and results saving folder
    model_folder = config.model_folder

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    #===== train the model for each folds
    metrics = {}
    model, dev_metrics, test_metrics, train_metrics = train_one_contrast(config = config, 
                    train_batches=clean_b, 
                    dev_batches = dev_b,
                    dev_insts = dev_insts,
                    test_batches = test_b,
                    test_insts = test_insts,
                    train_insts= train_insts)
    for mname in dev_metrics.keys():   
        metrics['dev_'+mname] = dev_metrics[mname]

    for mname in test_metrics.keys():   
        metrics['test_'+mname] = test_metrics[mname]  

    for mname in train_metrics.keys():   
        metrics['train_'+mname] = train_metrics[mname]      

    plotinnermetrics(metricslist = ['precision','recall','f1'], inner_metrics = metrics, info='before_total', conf = config)
    plotinnermetrics(metricslist = ['hard_precision','hard_recall','hard_f1'], inner_metrics = metrics,  info='before_hard', conf = config)
    plotinnermetrics(metricslist = ['easy_precision','easy_recall','easy_f1'], inner_metrics = metrics,  info='before_easy', conf = config)
    plotinnermetrics(metricslist = ['head_precision','head_recall','head_f1'], inner_metrics = metrics,  info='before_head', conf = config)
    plotinnermetrics(metricslist = ['tail_precision','tail_recall','tail_f1'], inner_metrics = metrics,  info='before_tail', conf = config)
    plotinnermetrics(metricslist = ['high_precision','high_recall','high_f1'], inner_metrics = metrics, info='before_highfreq', conf = config)
    plotinnermetrics(metricslist = ['low_precision','low_recall','low_f1'], inner_metrics = metrics,  info='before_lowfreq', conf = config)
    plotinnermetrics(metricslist = ['train_loss'], inner_metrics = metrics, info='before_loss', conf = config)
    plotinnermetrics(metricslist = ['train_noise_precision','train_noise_recall','train_noise_f1'], inner_metrics = metrics, info='noise', conf = config)
    plotinnermetrics(metricslist = ['train_clean_precision','train_clean_recall','train_clean_f1'], inner_metrics = metrics, info='clean', conf = config)
     
    #===== get the confidence score as out-of-sample prediction error
    print("\n\n[Data Info] Getting confidence score predicted using clean samples")
    train_insts.setids([all_train_ids[_in] for _in in index_rest])
    train_b =  batching_list_iterator(config, train_insts, data_collator)
    train_insts.setids(all_train_ids)
    # change the label of the other trainning fold 
    attach_useclean_score(config=config, model=model,
                    fold_batches = train_b,
                    folded_insts = train_insts,
                    folded_ids = index_rest)
    train_insts.setids([all_train_ids[_in] for _in in index_rest])    

    print('length of rest train: %d'%(len(train_insts)))           
    

def train_one_contrast(config: Config, 
              train_batches: List[dict], 
              dev_batches: List[dict],
              dev_insts: MapStyleJsonDataset,
              test_batches: List[dict],
              test_insts: MapStyleJsonDataset,
              train_insts: MapStyleJsonDataset,
              epoch = 30):  

    model = NNCRF_contrast(config) # initilize a new model
    # model = torch.nn.DataParallel(model) # !!! distributed 
    # model.to(config.device)
    epoch = config.cleanepochs

    model.train()
    if config.encoder =='bilstm':
        optimizer = get_optimizer(config, model)
    else:    
        optimizer, scheduler = get_huggingface_optimizer_and_scheduler(config, model, num_training_steps=len(train_batches) * epoch,
                                                                   weight_decay=0.0, eps = 1e-8, warmup_step=0)

    #epoch = config.num_epochs
    best_dev_f1 = -1 # save the best one
    allmetrics = ['loss']
    for metric in ['precision','recall','f1']:
        allmetrics.extend([kind+metric for kind in ['','hard_','easy_','high_','low_','head_','tail_']])
    
    test_metrics_list = {metric:[] for metric in allmetrics}
    dev_metrics_list = {metric:[] for metric in allmetrics} 
    
    for metric in ['precision','recall','f1']:
        allmetrics.extend([kind+metric for kind in ['noise_','clean_']])
    train_metrics_list = {metric:[] for metric in allmetrics}    

    for i in range(1, epoch + 1):
                
        epoch_loss = 0
        
        start_time = time.time()
        model.zero_grad()
        
        if config.encoder=='bilstm':
            optimizer = lr_decay(config, optimizer, i)
        
        for index in np.random.permutation(len(train_batches)):
            model.train()

            if i < config.alpha_warmup_ratio * epoch:
                alpha_t = 0.0
            else:
                alpha_t = config.alpha * (1.05**i)
            
            # input data and parameters to the forward function
            tmp = train_batches[index]
            
            # get the metrics: 
            loss= model(tmp, alpha_t)
            
            epoch_loss += loss.item()
            
            loss.backward()
            if config.encoder=='bilstm':
                optimizer.step()
            else:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            model.zero_grad()

        epoch_loss /= len(train_batches)
        
        train_metrics_list['loss'].append(epoch_loss)

        end_time = time.time()
        print("Epoch %d: loss %.5f, Time is %.2fs" % (i, epoch_loss, end_time - start_time), flush=True)
        
        #======== evaluate the model using dev and test data
        model.eval()
        train_metrics,_ = evaluate_model(config, model, train_batches, "train", train_insts)
        dev_metrics = evaluate_model(config, model, dev_batches, "dev", dev_insts)
        test_metrics = evaluate_model(config, model, test_batches, "test", test_insts)
        
        if dev_metrics['f1'] >= best_dev_f1:
            print("saving the best model so far for all epoches...")
            best_dev_f1 = dev_metrics['f1']
            best_model = copy.deepcopy(model)
            #torch.save(model.state_dict(), model_name)   

        model.zero_grad()
    
        for metric in dev_metrics:
            dev_metrics_list[metric].append(dev_metrics[metric])
        
        for metric in test_metrics:
            test_metrics_list[metric].append(test_metrics[metric])

        for metric in train_metrics:
            train_metrics_list[metric].append(train_metrics[metric])    
    
    print(f"The best dev F1: {best_dev_f1}") 

    return best_model, dev_metrics_list, test_metrics_list, train_metrics_list
