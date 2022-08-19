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
import higher

# from datastruct import Instance
from config import Config
# from datastruct import Reader, ContextEmb
from utils.utils import *
from utils.trans_utils import *
from evaluation.eval import *
from clmodel.baseline import NNCRF_baseline


def train_mw(config: Config, data_collator: Collator, 
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

    index_clean = random.sample(range(len(train_insts)), max(10, int(len(train_insts)*config.cleanprop)))
    index_rest = [i for i in range(len(train_insts)) if i not in index_clean]

    all_train_ids = train_insts.ids
    for i in index_clean:
        train_insts.update(i,'labels_id', train_insts[i]['gold_labels_id'])
        train_insts.update(i, 'labels', train_insts[i]['gold_labels'])
    train_insts.setids([all_train_ids[i] for i in index_clean])
    clean_batches = batching_list_iterator(config, train_insts, data_collator)

    train_insts.setids([all_train_ids[i] for i in index_rest])
    train_batches = batching_list_iterator(config, train_insts, data_collator)
    
    #==== to save the model with best dev f1 so far
    model_name = model_folder + "/"+"best_lstm_crf.m"
    config_name = model_folder + "/"+"best_config.conf"
    res_name = res_folder + "/"+"best_lstm_crf.results".format()

    #==== train the model
    #if not os.path.exists(model_name):
 
    train_metrics, dev_metrics, test_metrics, best_dev_f1 = train_one_mw(config = config, 
                            train_batches=train_batches, 
                            clean_batches=clean_batches,
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
    
    # print(str(metrics[iter]))
    # print("Archiving the best Model...")
    # with tarfile.open(model_folder + "/" + str(num_outer_iterations) + model_folder + ".tar.gz", "w:gz") as tar:
    #     tar.add(model_folder, arcname=os.path.basename(model_folder))
    # model = NNCRF_baseline(config)
    # model.load_state_dict(torch.load(model_name))
    # model.eval() 

    #====== evaluate the best model
    # best_test_metrics = evaluate_model(config, model, test_batches, "test", test_insts)
    # write_results(res_name, test_insts) 

    return metrics

def train_one_mw(config: Config, 
              train_batches: List[dict], 
              clean_batches:List[dict],
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
    model = NNCRF_baseline(config, reduce = False) 
    model.train()
    
    # get optimizer and lr rate scheduler
    if config.encoder =='bilstm':
        optimizer = get_optimizer(config, model)
    else:    
        optimizer, scheduler = get_huggingface_optimizer_and_scheduler(config, model, num_training_steps=len(train_batches) * epoch,
                                                                   weight_decay=0.0, eps = 1e-8, warmup_step=0)
        sgd_parameters = [
            {'params': [p for n, p in model.named_parameters()],
             'weight_decay': 0.0, 'lr': config.learning_rate}
        ]  
        meta_optimizer = torch.optim.SGD(sgd_parameters, lr=config.learning_rate*1e4)                                                         
        
    # training
    for i in range(1, epoch + 1):          
        epoch_loss = 0
        testepoch_loss = 0
        devepoch_loss = 0
        
        #======= 
        start_time = time.time()

        model.zero_grad()
        
        for index in np.random.permutation(len(train_batches)):
            model.train()
            index_clean = index % len(clean_batches)
            clean_batches_curr = clean_batches[index_clean]

            # to compute weights
            with torch.backends.cudnn.flags(enabled=False):
                with higher.innerloop_ctx(model, meta_optimizer) as (meta_model, meta_opt):
                    yf = meta_model(train_batches[index])
                    eps = torch.zeros(yf.size(), requires_grad=True, device=model.device) # initialize all from zero
                    meta_train_loss = torch.sum(yf * eps)
                    meta_opt.step(meta_train_loss)  # differentiable optimizer

                    yg = meta_model(clean_batches_curr)
                    meta_val_loss = torch.mean(yg)
                    grad_eps = torch.autograd.grad(meta_val_loss, eps, allow_unused=True)[0].detach()
                    del meta_opt
                    del meta_model
           
            w_tilde = torch.sigmoid(-grad_eps+torch.mean(grad_eps))
            norm_w = torch.sum(w_tilde)
            if norm_w != 0:
                w = w_tilde / norm_w
            else:
                w = w_tilde
            if index==0:   
                print(grad_eps)   
                print(w_tilde)  
                print(w)  

            # to update model parameter    

            yf = model(train_batches[index])
            if index==0:   
                print(yf)
            batch_loss = torch.sum(yf * w)
            model.zero_grad()
            batch_loss.backward()
            loss_val = batch_loss.data.item()
            epoch_loss += loss_val  

            # nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
            #                          max_norm=config.grad_clip)
            optimizer.step() 
            optimizer.zero_grad() 
            scheduler.step()

        # update weight parameter
        epoch_loss /= len(train_batches)

        end_time = time.time()
        print("Epoch %d: loss %.5f, Time is %.2fs" % (i, epoch_loss, end_time - start_time), flush=True)
    
        train_metrics_list['loss'].append(epoch_loss)
        
        #======== evaluate the model using dev and test data
        model.eval()

        for index in np.random.permutation(len(test_batches)):
            
            # input data and parameters to the forward function
            # get the metrics: 
            testloss = model(test_batches[index], train=False)
            testepoch_loss += torch.mean(testloss).item()   
        
        testepoch_loss /= len(test_batches)

        test_metrics_list['loss'].append(testepoch_loss)

        for index in np.random.permutation(len(dev_batches)):
            
            # input data and parameters to the forward function
            # get the metrics: 
            devloss = model(dev_batches[index], train=False)
            devepoch_loss += torch.mean(devloss).item()  
        
        devepoch_loss /= len(dev_batches)

        dev_metrics_list['loss'].append(devepoch_loss)

        #====== get other metrics
        # metric is [precision, recall, f_score]
        if config.diagonosis and train_insts is not None:
            train_metrics,_,_,_ = evaluate_model(config, model, train_batches, "train", train_insts)
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

