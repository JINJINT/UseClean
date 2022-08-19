# from turtle import pos
from operator import neg
import numpy as np
from requests import head
from overrides import overrides
from typing import List, Tuple, Dict, Union, Set
from config import Config
from datastruct import Span
import torch
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from utils.utils import *
from evaluation.fitmix import *
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv

def ratio_estimation_iterator(config, insts, indexs=None):
    """
    estimate the rate of positive noise and negative noise in the whole training dataset
    """
    neg_total=0
    pos_total=0
    neg_noise=0
    pos_noise=0
    O_index=config.label2idx['O']
    
    if indexs is None:
        indexs = range(len(insts))

    for i in indexs:
        inst = insts[i]
        for n1,n2 in zip(inst["labels_id"], inst["gold_labels_id"]):
            if(n1==O_index):
                neg_total+=1
                if(n1!=n2):
                    neg_noise+=1
            else:
                pos_total+=1
                if(n1!=n2): 
                    pos_noise+=1
    neg_noise_rate=float(neg_noise)/(float(neg_total)+1e-8)
    pos_noise_rate=float(pos_noise)/(float(pos_total)+1e-8)
    
    return neg_noise_rate,pos_noise_rate


def evaluate_confscore(gold_tags, scores, tags, mask, negative_mask, batch_size, word_seq_lens, device, fitmix=False):
    #=========== check the fitness of confidence score of the clean_mask

    # whether the word-tag is entity or not
    positive_mask=(1-negative_mask)*mask
    positive_mask = positive_mask.bool().tolist()
    negative_mask = negative_mask.bool().tolist()

    # whether the word-tag is clean or not
    clean_mask=torch.eq(gold_tags,tags).float()*mask
    noise_mask=(1-clean_mask)*mask.float()
    noise_mask = noise_mask.tolist()

    # remove padding
    scores = np.array(scores).reshape(batch_size,-1)
    scores_removepadding = []
    for i,score in enumerate(scores):
        scores_removepadding.append(score[:word_seq_lens[i]].tolist())
    flat_scores_removepadding = np.array(flatten(scores_removepadding))

    #print(noise_mask)
    noise_removepadding = []
    for i,noise in enumerate(noise_mask):
        noise_removepadding.append(noise[:word_seq_lens[i]])
    flat_noise_removepadding = np.array(flatten(noise_removepadding))

    positive_removepadding = []
    for i,positive in enumerate(positive_mask):
        positive_removepadding.append(positive[:word_seq_lens[i]])
    flat_positive_removepadding = flatten(positive_removepadding)
    #print(flat_positive_removepadding)

    negative_removepadding = []
    for i,negative in enumerate(negative_mask):
        negative_removepadding.append(negative[:word_seq_lens[i]])
    flat_negative_removepadding = flatten(negative_removepadding)
    
    if len(list(set(flat_noise_removepadding[flat_negative_removepadding])))>1:
        neg_logis_model = LogisticRegression(solver='liblinear', random_state=0)
        neg_logis_model.fit(flat_scores_removepadding[flat_negative_removepadding].reshape(-1,1), flat_noise_removepadding[flat_negative_removepadding])
        #negfitscore = neg_logis_model.score(flat_scores_removepadding[flat_negative_removepadding].reshape(-1,1), flat_noise_removepadding[flat_negative_removepadding])
        negpredict = neg_logis_model.predict(flat_scores_removepadding[flat_negative_removepadding].reshape(-1,1))
        #negfitscore = f1_score(flat_noise_removepadding[flat_negative_removepadding], negpredict)
        neg_predict = np.sum(negpredict)
        neg_p = np.sum([p*q for p,q in zip(negpredict, flat_noise_removepadding[flat_negative_removepadding])])
    else:
        #negfitscore = 1
        negpredict = flat_noise_removepadding[flat_negative_removepadding]
        neg_predict = np.sum(negpredict)
        neg_p = np.sum(negpredict)

    if len(list(set(flat_noise_removepadding[flat_positive_removepadding])))>1:     
        pos_logis_model = LogisticRegression(solver='liblinear', random_state=0)
        pos_logis_model.fit(flat_scores_removepadding[flat_positive_removepadding].reshape(-1,1), flat_noise_removepadding[flat_positive_removepadding])
        #posfitscore = pos_logis_model.score(flat_scores_removepadding[flat_positive_removepadding].reshape(-1,1), flat_noise_removepadding[flat_positive_removepadding])
        pospredict = pos_logis_model.predict(flat_scores_removepadding[flat_positive_removepadding].reshape(-1,1))
        #posfitscore = f1_score(flat_noise_removepadding[flat_positive_removepadding], pospredict)
        pos_predict = np.sum(pospredict)
        pos_p = np.sum([p*q for p,q in zip(pospredict, flat_noise_removepadding[flat_positive_removepadding])])
    else:
        #posfitscore = 1
        pospredict = flat_noise_removepadding[flat_positive_removepadding]
        pos_predict = np.sum(pospredict)
        pos_p = np.sum(pospredict)

    if fitmix: # use fit mixture of gamma and gaussian
        negscore = np.array(flat_scores_removepadding[flat_negative_removepadding])
        negcutoff = get_cutoff(negscore, thred = 0.1, recall=0.9)
        negpredict = (negscore<negcutoff)

        posscore = np.array(flat_scores_removepadding[flat_positive_removepadding])
        poscutoff = get_cutoff(posscore, thred = 0, recall=0.7)
        pospredict = (posscore<poscutoff)

    allpredict = flat_noise_removepadding
    allpredict[flat_positive_removepadding] = pospredict.copy()
    allpredict[flat_negative_removepadding] = negpredict.copy()
    
    neg_remove_rate = 1-np.mean(negpredict)
    pos_remove_rate = 1-np.mean(pospredict)

    # adding the paddings back
    predicted_mask = torch.zeros_like(clean_mask)
    # print(predicted_mask)
    cur=0
    for i, pre in enumerate(predicted_mask):
        pre[:word_seq_lens[i]] = 1-torch.tensor(list(allpredict[cur: (cur+word_seq_lens[i])].copy())).to(device) # to remember
        cur+=word_seq_lens[i]    

    # print(predicted_mask)
    #results = {"fitscore": (posfitscore+negfitscore)/2*100, 'neg_fitscore': negfitscore*100, 'pos_fitscore': posfitscore*100}
    results = {'pos_fit_predict': pos_predict, 'neg_fit_predict': neg_predict, 'pos_fit_p': pos_p, 'neg_fit_p': neg_p}
    
    return results, predicted_mask


def evaluate_conf(gold_tags, tags,  small_loss_mask, mask, negative_mask):
    """
       mask: whether the word-tag is padding or not
       small_loss_mask: whether the word-tag is keeped by the alg or not
       negative_mask: whether the tag is entity or not
    """

    # whether the word-tag is removed by the algorithms or not
    remove_mask=(1-small_loss_mask)*mask
    
    # whether the word-tag is entity or not
    positive_mask=(1-negative_mask)*mask
    
    # whether the word-tag is clean or not
    clean_mask=torch.eq(gold_tags,tags).float()*mask
    noise_mask=(1-clean_mask)*mask.float()

    #============= compute the sliced version by nonentity versus entity
    
    #==== count correctly removed noisy nonentity and entity
    remove_neg=remove_mask*negative_mask
    remove_right_neg=noise_mask*remove_neg
    
    remove_pos=remove_mask*(1-negative_mask)*mask
    remove_right_pos=noise_mask*remove_pos
    
    #==== count actual noisy nonentity and entity
    noise_positive=noise_mask*positive_mask*mask
    noise_negative=noise_mask*negative_mask*mask
    
    #==== summarize
    neg_entity = noise_negative.sum().item()
    neg_predict = remove_neg.sum().item()
    neg_p = remove_right_neg.sum().item()

    pos_entity = noise_positive.sum().item()
    pos_predict = remove_pos.sum().item()
    pos_p = remove_right_pos.sum().item()

    results = {'neg_entity': neg_entity,
               'neg_predict': neg_predict,
               'neg_p': neg_p,
               'pos_entity': pos_entity,
               'pos_predict': pos_predict,
               'pos_p': pos_p,
               }

    return results




def evaluate_model(config: Config, model, batched_data, name: str, list_data, epoch=None):
    '''
    model: trained ner model
    batch_insts_ids: batch id
    name: str
    batched_data: list of dictionary, each dictionary contains one batch
    '''
    # evaluation
    allmetrics = None
    allscores = {}
    batch_id = 0
    batch_size = config.batch_size

    # for studying the distribution of confidence score
    # we collect the following over all the training samples
    alliscleanlist = []
    allisotherlist = []
    allscorelist = []
    allwordslist = []
    allentitylist = []
    allgoldentitylist = []

    neg_remove_rate, pos_remove_rate = None, None

    for batch in batched_data:        
        # get the score and id 
        batch_max_scores, batch_max_ids = model.decode(batch)
        if epoch is not None:
            if config.score in [ 'nerloss','encoderloss','diff','spike','entropy',"aum"]:
                if config.cutoff in ['goracle','fake','fitmix']:   
                    plotscore = True # collect all score for fitting every epoches
                else:    
                    plotscore = (epoch<=10 or (epoch % 5==0)) # collect all score for plot every 5 epoches
            else:
                plotscore = (epoch == 1) # collect all scores for finding cutoff globally
        else:
            plotscore = False    

        hardentity = []
        easyentity = []
        highfreqentity = []
        lowfreqentity = []
        headentity = []
        tailentity = []

        # the following function also attach prediction to batch_insts
        if config.dataset.startswith('massive'):
            hardentity = set(['artist_name', 'person', 'date', 'time', 'event_name', 'list_name'])
            easyentity = set(['music_album', 'transport_name', 'transport_descriptor', 'email_address', 'email_folder'])
            highfreqentity = set(['date', 'place_name', 'event_name', 'person', 'time', 'media_type', 'business_name', 'weather_descriptor', 'transport_type', 'food_type'])
            lowfreqentity = set(['personal_info','meal_type', 'playlist_name','podcast_name','time_zone', 
                                'app_name', 'change_amount', 'music_descriptor','joke_type', 'transport_agency',
                                'email_address', 'email_folder', 'ingredient', 'coffee_type', 'cooking_type', 
                                'movie_name', 'transport_name', 'alarm_type', 'movie_type', 'drink_type', 'audiobook_author', 
                                'transport_descriptor', 'sport_type', 'music_album', 'game_type'])
            headentity = set(['date', 'time'])
            tailentity = set(['relation', 'timeofday', 'house_place', 'music_genre', 'business_type', 'player_setting', 'audiobook_name', 'game_name', 
                            'podcast_descriptor', 'email_address', 'general_frequency', 'playlist_name', 'podcast_name', 'order_type', 'personal_info', 
                            'color_type', 'change_amount', 'time_zone', 'music_descriptor', 'meal_type', 'app_name', 'joke_type', 'transport_agency', 'movie_name', 
                            'coffee_type', 'ingredient', 'email_folder', 'transport_name', 'alarm_type', 'cooking_type', 'movie_type', 'audiobook_author', 'transport_descriptor', 'drink_type', 'sport_type', 'music_album', 'game_type'])

        if config.dataset.startswith('conll') or config.dataset.startswith('eng') or config.dataset.startswith('wiki'):
            hardentity = ['MISC']
            easyentity = ['PER']
            highfreqentity = ['ORG']
            lowfreqentity = ['LOC']
            headentity = ['PER','ORG']
            tailentity = ['MISC','LOC']

        metrics, iscleanlist, isotherlist, scorelist, entitylist, goldentitylist, wordslist = evaluate_batch_insts(batch_id=batch_id,
                            batched_data = batch,
                            listed_data = list_data,
                            batch_size = config.batch_size,
                            batch_pred_ids = batch_max_ids,
                            idx2label = config.idx2label,
                            noise = (name=='train'),
                            plotscore = plotscore,
                            hardentity = hardentity,
                            easyentity = easyentity,
                            highfreqentity = highfreqentity,
                            lowfreqentity = lowfreqentity,
                            headentity = headentity,
                            tailentity = tailentity) # plotscore indicate whether plot the distribution of confidence score 
        
        allwordslist.extend(wordslist)
        allentitylist.extend(entitylist)
        allgoldentitylist.extend(goldentitylist)
        alliscleanlist.extend(iscleanlist)
        allisotherlist.extend(isotherlist)
        allscorelist.extend(scorelist)    
        
        if allmetrics is None:
            allmetrics = metrics.copy()

        for metricsname in allmetrics.keys():
            for metr in allmetrics[metricsname].keys():
                allmetrics[metricsname][metr]+=metrics[metricsname][metr]

        batch_id += 1

    for metricsname in metrics.keys():
        p, total_predict, total_entity = allmetrics[metricsname]['p'], allmetrics[metricsname]['predict'], allmetrics[metricsname]['entity']
        precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
        recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
        fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
        if metricsname=='total':
            metricsnames = ''
        else:
            metricsnames = metricsname+'_'    
        allscores[metricsnames+'precision'] = precision
        allscores[metricsnames+'recall'] = recall
        allscores[metricsnames+'f1'] = fscore
        if metricsname =='total':
            print("[%s set] %s Precision: %.2f, Recall: %.2f, F1: %.2f" % (name, metricsname, precision, recall, fscore), flush=True)
    
    allpredict=None      
    if plotscore: # having collected confscores, only for training dataset
        fullpath = config.res_folder
        if config.diagonosis:
            filename = fullpath + '/confscore_'+str(epoch)+'.txt'
        else:
            filename = fullpath + '/confscore.txt'    
        
        if not os.path.exists(fullpath):
            os.makedirs(fullpath) 

        with open(filename, 'w') as f:
            for i in range(len(allscorelist)):
                f.write("%s %s %s %d %d %.2f" % (allwordslist[i], allentitylist[i], allgoldentitylist[i], alliscleanlist[i], allisotherlist[i], allscorelist[i]))
                f.write("\n") 
        
        scoredata = pd.read_csv(filename, sep=" ", header=None, quoting=csv.QUOTE_NONE)
        scoredata.columns = ["words", "entity", "goldentity", "isclean", "isother", "score"]

        if config.diagonosis:
            plt.figure(figsize=(5,3))
            sns.histplot(data=scoredata, x="score", hue="isclean")
            plt.savefig(config.res_folder+'/confscore_'+str(epoch)+'.png')
            plt.close()  
            
            posscoredata = scoredata[scoredata['isother']==0]
            plt.figure(figsize=(5,3))
            sns.histplot(data=posscoredata, x="score", hue="isclean")
            plt.savefig(config.res_folder+'/confscore_pos_'+str(epoch)+'.png')
            plt.close()  

            negscoredata = scoredata[scoredata['isother']==1]
            plt.figure(figsize=(5,3))
            sns.histplot(data=negscoredata, x="score", hue="isclean")
            plt.savefig(config.res_folder+'/confscore_neg_'+str(epoch)+'.png')
            plt.close()
            
        # if config.cutoff =='goracle':
        #     if (epoch is not None and epoch==1 and config.score in ['cross','useclean','usecleanhead','usecleantail']) or (config.score in ['nerloss','encoderloss','diff','spike','entropy','aum']):
        #         negscore = np.array(scoredata[scoredata['isother']==1]['score'])   
        #         negres = np.array(scoredata[scoredata['isother']==1]['isclean'])
        #         if len(list(set(negres)))>1:     
        #             neg_logis_model = LogisticRegression(solver='liblinear', random_state=0)
        #             neg_logis_model.fit(negscore.reshape(-1,1), negres)
        #             negpredict = neg_logis_model.predict(negscore.reshape(-1,1))
        #         else:
        #             negpredict = negres
        #         neg_remove_rate = 1- np.mean(negpredict)
        #         config.negcutoff = np.quantile(negscore, 1-neg_remove_rate)

        #         posscore = np.array(scoredata[scoredata['isother']==0]['score'])  
        #         posres = np.array(scoredata[scoredata['isother']==0]['isclean'])
        #         if len(list(set(posres)))>1:     
        #             pos_logis_model = LogisticRegression(solver='liblinear', random_state=0)
        #             pos_logis_model.fit(posscore.reshape(-1,1), posres)
        #             pospredict = pos_logis_model.predict(posscore.reshape(-1,1))
        #         else:
        #             pospredict = posres

        #         pos_remove_rate = 1-np.mean(pospredict)
        #         config.poscutoff = np.quantile(posscore, 1-pos_remove_rate)
                
        #         allpredict = np.array(scoredata['isclean'].copy())
        #         allpredict[scoredata['isother']==1] = negpredict
        #         allpredict[scoredata['isother']==0] = pospredict
        #     else:
        #         allpredict = None             
        
        if config.cutoff in ['fake','fitmix','goracle']: 
            if (epoch is not None and epoch==1 and config.score in ['cross','useclean','usecleanhead','usecleantail']) or (config.score in ['nerloss','encoderloss','diff','spike','entropy','aum']):           
                
                posscore = np.array(scoredata[scoredata['isother']==0]['score'])
                pospredict = (posscore < config.poscutoff)
                
                negscore = np.array(scoredata[scoredata['isother']==1]['score'])
                negpredict = (negscore < config.negcutoff) 
  
                allpredict = np.array(scoredata['isclean'].copy())
                allpredict[scoredata['isother']==1] = negpredict
                allpredict[scoredata['isother']==0] = pospredict 


        # if config.cutoff =='fitmix':
        #     if (epoch is not None and epoch==1 and config.score in ['cross','useclean','usecleanhead','usecleantail']) or (config.score in ['nerloss','encoderloss','diff','spike','entropy','aum']):
        #         negscore = np.array(scoredata[scoredata['isother']==1]['score'])
        #         if config.usecleanscore=='nerloss':
        #             thred = 0.1
        #         else:
        #             thred = 0    
        #         negcutoff = get_cutoff(negscore, thred = thred, recall=config.recall, usef1=config.usef1)
        #         if config.dataset.startswith('massive_en_us__noise_miss') or config.dataset.startswith('conll_noise_miss') or config.dataset in ['conll_transfer', 'conll_dist', 'massive_transeasy', 'massive_dist','wikigold_dist','wikigold_self']:
        #             negpredict = (negscore<negcutoff)
        #         else:
        #             negpredict = (negscore>=0)
        #             negcutoff = 10000
        #         neg_remove_rate = 1- np.mean(negpredict)
        #         config.negcutoff = negcutoff
                
        #         posscore = np.array(scoredata[scoredata['isother']==0]['score'])
        #         poscutoff = get_cutoff(posscore, thred = 0, recall=config.recall, usef1=config.usef1)
        #         pospredict = (posscore<poscutoff)
        #         pos_remove_rate = 1-np.mean(pospredict)
        #         config.poscutoff = poscutoff
                
        #         allpredict = np.array(scoredata['isclean'].copy())
        #         allpredict[scoredata['isother']==1] = negpredict
        #         allpredict[scoredata['isother']==0] = pospredict
        #     else:
        #         allpredict = None             

        
    if name=='train':
        if neg_remove_rate is not None and pos_remove_rate is not None:
            return allscores, allpredict, neg_remove_rate, pos_remove_rate
        else:
            return allscores, allpredict, None, None    
    else:
        return allscores


def evaluate_batch_insts(batch_id, 
                         batched_data,
                         listed_data,
                         batch_size,
                         batch_pred_ids: torch.Tensor,
                         idx2label={},
                         noise = False,
                         compall= True,
                         plotscore = False,
                         hardentity = set(),
                         easyentity = set(),
                         highfreqentity = set(),
                         lowfreqentity = set(),
                         headentity = set(),
                         tailentity = set()
                         ) -> np.ndarray:


    """
    Evaluate a batch of instances and handling the padding positions.
    :param batch_insts:  a batched of instances.
    :param batch_pred_ids: Shape: (batch_size, max_length) prediction ids from the viterbi algorithm.
    :param batch_gold_ids: Shape: (batch_size, max_length) gold ids.
    :param word_seq_lens: Shape: (batch_size) the length for each instance.
    :param idx2label: The idx to label mapping.
    :return: numpy array containing (number of true positive, 
                                     number of all positive, 
                                     number of true positive + number of false negative)
             You can also refer as (number of correctly predicted entities, 
                                    number of entities predicted, 
                                    number of entities in the dataset)
    """

    if compall:
        if noise:
            allmetrics = {metric: {'p':0, 'entity':0, 'predict':0} for metric in ['total','hard','easy','high','low','head','tail','noise','clean']}
        else:    
            allmetrics = {metric: {'p':0, 'entity':0, 'predict':0} for metric in ['total','hard','easy','high','low','head','tail']}
    else: 
        allmetrics = {metric: {'p':0, 'entity':0, 'predict':0} for metric in ['total']}

    word_seq_lens = batched_data['word_seq_lens'].tolist()
    batch_gold_ids = batched_data['gold_labels_id']

    iscleanlist = []
    isotherlist = []
    scorelist = [] 
    entitylist= []
    wordslist = []
    goldentitylist = [] 
    
    for idx in range(len(batch_pred_ids)):
        list_idx = batch_id * batch_size + idx
        # remove the padding
        length = word_seq_lens[idx]
        output = batch_gold_ids[idx][:length].tolist()
        prediction = batch_pred_ids[idx][:length].tolist()
        
        # this is the right way!!
        prediction = prediction[::-1]
        
        # convert the token to label
        output = [idx2label[l] for l in output]
        prediction =[idx2label[l] for l in prediction]
        
        
        #===== assign info
        # also give the prediction to batch_inst
        listed_data.update(list_idx, 'prediction', prediction)
        # also give the is_clean_predict
        
        #===== extract info 
        listed_data_cur = listed_data[list_idx]
        if noise and plotscore:
            scorelist.extend(listed_data_cur['scores'])

        # if applying on train set, then also extract the is_noisy label
        if noise:
            is_clean = listed_data_cur['isclean']  
            if plotscore:
                iscleanlist.extend(is_clean)
                isotherlist.extend(listed_data_cur['isother'])
                goldentitylist.extend(listed_data_cur['gold_labels'])
                entitylist.extend(listed_data_cur['labels'])
                wordslist.extend(listed_data_cur['words'])

        output_spans = set()
        start = -1
        if compall:
            hard_output_spans = set()
            high_output_spans = set()
            noisy_output_spans = set()
            easy_output_spans = set()
            low_output_spans = set()
            clean_output_spans = set()
            head_output_spans = set()
            tail_output_spans = set()

        for i in range(len(output)):
            if output[i].startswith("B-"):
                start = i
            if output[i].startswith("E-"):
                end = i
                output_spans.add(Span(start, end, output[i][2:])) 
                if compall:
                    if output[i][2:] in hardentity:
                        hard_output_spans.add(Span(start, end, output[i][2:]))
                    if output[i][2:] in easyentity:
                        easy_output_spans.add(Span(start, end, output[i][2:]))    
                    
                    if output[i][2:] in highfreqentity:  
                        high_output_spans.add(Span(start, end, output[i][2:]))  
                    if output[i][2] in lowfreqentity:
                        low_output_spans.add(Span(start, end, output[i][2:])) 

                    if output[i][2:] in headentity:  
                        head_output_spans.add(Span(start, end, output[i][2:]))  
                    if output[i][2] in tailentity:
                        tail_output_spans.add(Span(start, end, output[i][2:]))     

                    if noise and is_clean is not None:
                        if all(is_clean[start:(end+1)]):
                            clean_output_spans.add(Span(start, end, output[i][2:])) 
                        else:
                            noisy_output_spans.add(Span(start, end, output[i][2:]))      
            
            if output[i].startswith("S-"):
                output_spans.add(Span(i, i, output[i][2:]))  
                if compall:
                    if output[i][2:] in hardentity:
                        hard_output_spans.add(Span(i, i, output[i][2:]))
                    if output[i][2:] in easyentity:
                        easy_output_spans.add(Span(i, i, output[i][2:]))    
                    
                    if output[i][2:] in highfreqentity:  
                        high_output_spans.add(Span(i, i, output[i][2:]))
                    if output[i][2:] in lowfreqentity:
                        low_output_spans.add(Span(i, i, output[i][2:]))   

                    if output[i][2:] in headentity:  
                        head_output_spans.add(Span(i, i, output[i][2:]))
                    if output[i][2:] in tailentity:
                        tail_output_spans.add(Span(i, i, output[i][2:]))            

                    if noise and is_clean is not None:
                        if is_clean[i]:
                            clean_output_spans.add(Span(i, i, output[i][2:])) 
                        else:
                            noisy_output_spans.add(Span(i, i, output[i][2:])) 
                               
        
        predict_spans = set()
        if compall:
            hard_pred_spans = set()
            high_pred_spans = set()
            noisy_pred_spans = set()
            easy_pred_spans = set()
            low_pred_spans = set()
            clean_pred_spans = set()
            head_pred_spans = set()
            tail_pred_spans = set()
        for i in range(len(prediction)):
            if prediction[i].startswith("B-"):
                start = i
            if prediction[i].startswith("E-"):
                end = i
                predict_spans.add(Span(start, end, prediction[i][2:]))
                if all:
                    if prediction[i][2:] in hardentity:
                        hard_pred_spans.add(Span(start, end, prediction[i][2:]))
                    if prediction[i][2:] in easyentity:
                        easy_pred_spans.add(Span(start, end, prediction[i][2:]))    
                    
                    if prediction[i][2:] in highfreqentity:  
                        high_pred_spans.add(Span(start, end, prediction[i][2:]))  
                    if prediction[i][2:] in lowfreqentity:  
                        low_pred_spans.add(Span(start, end, prediction[i][2:]))  
                    
                    if prediction[i][2:] in headentity:  
                        head_pred_spans.add(Span(start, end, prediction[i][2:]))  
                    if prediction[i][2:] in tailentity:  
                        tail_pred_spans.add(Span(start, end, prediction[i][2:]))

                    if noise and is_clean is not None:
                        if all(is_clean[start:(end+1)]):
                            clean_pred_spans.add(Span(start, end, prediction[i][2:])) 
                        else:
                            noisy_pred_spans.add(Span(start, end, prediction[i][2:]))     

            if prediction[i].startswith("S-"):
                predict_spans.add(Span(i, i, prediction[i][2:]))
                if compall:
                    if prediction[i][2:] in hardentity:
                        hard_pred_spans.add(Span(i, i, prediction[i][2:]))
                    if prediction[i][2:] in easyentity:
                        easy_pred_spans.add(Span(i, i, prediction[i][2:]))    
                    
                    if prediction[i][2:] in highfreqentity:  
                        high_pred_spans.add(Span(i, i, prediction[i][2:]))
                    if prediction[i][2:] in lowfreqentity:
                        low_pred_spans.add(Span(i, i, prediction[i][2:])) 

                    if prediction[i][2:] in headentity:  
                        head_pred_spans.add(Span(i, i, prediction[i][2:]))
                    if prediction[i][2:] in tailentity:
                        tail_pred_spans.add(Span(i, i, prediction[i][2:]))        
                    
                    if noise and is_clean is not None:
                        if is_clean[i]:
                            clean_pred_spans.add(Span(i, i, prediction[i][2:])) 
                        else:
                            noisy_pred_spans.add(Span(i, i, prediction[i][2:]))      
        
        #====get the total counts
        allmetrics['total']['entity'] += len(output_spans)
        allmetrics['total']['predict'] += len(predict_spans)
        allmetrics['total']['p'] += len(predict_spans.intersection(output_spans))
        
        if compall:
            #===get the sliced verison of counts
            # sliced by entity difficulty: below are easy to mix entities found in the data
            allmetrics['hard']['entity'] += len(hard_output_spans)
            allmetrics['hard']['predict'] += len(hard_pred_spans)
            allmetrics['hard']['p'] += len(hard_pred_spans.intersection(hard_output_spans))

            allmetrics['easy']['entity'] += len(easy_output_spans)
            allmetrics['easy']['predict'] += len(easy_pred_spans)
            allmetrics['easy']['p'] += len(easy_pred_spans.intersection(easy_output_spans))

            # sliced by entity frequency: below are highfrequent entities found in the data
            allmetrics['high']['entity'] += len(high_output_spans)
            allmetrics['high']['predict'] += len(high_pred_spans)
            allmetrics['high']['p'] += len(high_pred_spans.intersection(high_output_spans))

            allmetrics['low']['entity'] += len(low_output_spans)
            allmetrics['low']['predict'] += len(low_pred_spans)
            allmetrics['low']['p'] += len(low_pred_spans.intersection(low_output_spans))
            
            # sliced by entity tail or hear: below are head and tail entities found in the data
            allmetrics['head']['entity'] += len(head_output_spans)
            allmetrics['head']['predict'] += len(head_pred_spans)
            allmetrics['head']['p'] += len(head_pred_spans.intersection(head_output_spans))

            allmetrics['tail']['entity'] += len(tail_output_spans)
            allmetrics['tail']['predict'] += len(tail_pred_spans)
            allmetrics['tail']['p'] += len(tail_pred_spans.intersection(tail_output_spans))

            # sliced by entity noisy or not:
            if noise:
                allmetrics['noise']['entity'] += len(noisy_output_spans)
                allmetrics['noise']['predict'] += len(noisy_pred_spans)
                allmetrics['noise']['p'] += len(noisy_pred_spans.intersection(noisy_output_spans))

                allmetrics['clean']['entity'] += len(clean_output_spans)
                allmetrics['clean']['predict'] += len(clean_pred_spans)
                allmetrics['clean']['p'] += len(clean_pred_spans.intersection(clean_output_spans))

    return allmetrics, iscleanlist, isotherlist, scorelist, entitylist, goldentitylist, wordslist
