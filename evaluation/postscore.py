import numpy as np
from overrides import overrides
from typing import List, Tuple, Dict, Union, Set
from config import Config
from datastruct import Span
from datastruct.dataset import *
import torch
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from utils.utils import *
from torch.distributions import Categorical
import os
from evaluation.fitmix import *
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def attach_useclean_score(config: Config, model, 
                            fold_batches: List[dict],
                            folded_insts: MapStyleJsonDataset,
                            folded_ids: List[int]):
    """
    assign the confidence scores estimated by useclean method to the training data
    param
        fold_batches: the batched training data
        folded_insts: the training data of the iterator format 
        folded_ids: list of ids which contains which training samples to use in the final stage training (i.e. excluding those in small gold data)
    return
        no return, modify the folded_insts and config on the fly, 
        will save the confidence score, weights, fitted fake cutoff etc.   
    """
    batch_id = 0
    batch_size = config.batch_size

    if config.cutoff=='fake':
        # if using cutfake for cutoff fitting, then we will sample a few tokens and assign them with a "fake" class
        # we do this for entities and nonentities separately
        index_fake = random.sample(range(len(folded_ids)), config.numfake*int(len(folded_ids)/(config.label_size)))
        config.index_fake = index_fake
        oldids = {}
        oldlabels = {}
        for k in index_fake:
            i = folded_ids[k]
            oldids[i] = folded_insts[i]['labels_id']
            oldlabels[i] = folded_insts[i]['labels'] 
            newids = [config.label_size-2 if config.idx2label[e]!='O' else config.label_size-1 for e in folded_insts[i]['labels_id']]
            newlabs = ["fakepos" if e!="O" else "fakeneg" for e in folded_insts[i]['labels']]
            folded_insts.update(i,'labels_id', newids)
            folded_insts.update(i, 'labels', newlabs)

    model.eval()
    for batch in fold_batches:
        encoder_scores,_= model.encoder(batch['input_ids'], batch['word_seq_lens'], batch['attention_mask'], batch['token2word_mask'])
        onehot_label = torch.zeros_like(encoder_scores).scatter_(-1, batch['labels_id'].unsqueeze(-1), 1) 
        
        ner_loss = model.inferencer.marginal(encoder_scores, batch['word_seq_lens'])
        nerloss_confscore = -(ner_loss * onehot_label).sum(dim=-1).detach().cpu().numpy()
        
        log_token_prob = torch.nn.functional.log_softmax(encoder_scores, dim=-1)
        encoder_confscore = -(onehot_label * log_token_prob).sum(dim=-1).detach().cpu().numpy()

        maxscore = torch.max(encoder_scores, dim=-1).values
        obsscore = (encoder_scores * onehot_label).sum(dim=-1)
        diff_confscore = (maxscore-obsscore).detach().cpu().numpy() 

        masked_logits = encoder_scores + onehot_label*(-100000.0)
        max2score = torch.max(masked_logits, dim=-1).values
        obsscore = (encoder_scores * onehot_label).sum(dim=-1)
        aum_confscore = (obsscore-max2score).detach().cpu().numpy() 

        token_prob = torch.nn.functional.softmax(encoder_scores, dim=-1)
        token_entropy = Categorical(token_prob).entropy()
        entropy_confscore = token_entropy.detach().cpu().numpy() 

        spikescore = torch.max(token_prob, dim=-1).values
        spikeness_confscore = spikescore.detach().cpu().numpy()

        marginals = model.inferencer.marginal(encoder_scores, batch['word_seq_lens'])
        marginals = marginals.detach().cpu().numpy()
        word_seq_lens = batch['word_seq_lens'].cpu().numpy()
    
        word_seq_lens = batch['word_seq_lens'].cpu().numpy()
        for idx in range(len(word_seq_lens)):
            length = word_seq_lens[idx]
            fold_inst_idx = batch_id * batch_size + idx
            inst_idx = folded_ids[fold_inst_idx]
            folded_insts.update(inst_idx, 'nerloss', nerloss_confscore[idx,:length])
            folded_insts.update(inst_idx, 'encoderloss', encoder_confscore[idx,:length])
            folded_insts.update(inst_idx, 'diff', diff_confscore[idx,:length])
            folded_insts.update(inst_idx, 'entropy', entropy_confscore[idx,:length])
            folded_insts.update(inst_idx, 'spike', spikeness_confscore[idx,:length])
            folded_insts.update(inst_idx, 'aum', aum_confscore[idx,:length])
            folded_insts.update(inst_idx, 'scores', folded_insts[inst_idx][config.usecleanscore])
            folded_insts.update(inst_idx, 'weights', marginals[idx, :length, :])
        batch_id += 1

    
    # we will using the following function to compute the corresponding cutoff for entity and nonentities
    get_cut_useclean(config, fold_batches, folded_insts, folded_ids)
    
    if config.cutoff=='fake':     
        # re-assign the fake cases to their original labels after fitting the cutoff if using the "fake" method
        for k in index_fake:
            i = folded_ids[k]
            folded_insts.update(i,'labels_id', oldids[i])
            folded_insts.update(i, 'labels', oldlabels[i])  




def get_cut_useclean(config: Config, batched_data, list_data, folded_ids):

    allisotherlist = []
    allentitylist = []
    allscorelist = []
    alliscleanlist = []

    batch_id = 0
    for batch in batched_data:
        isotherlist = []
        entitylist = []
        scorelist = [] 
        iscleanlist = []
        for idx in range(len(batch['word_seq_lens'])):
            list_idx = batch_id * config.batch_size + idx
            list_idx = folded_ids[list_idx]
            listed_data_cur = list_data[list_idx]
            scorelist.extend(listed_data_cur['scores'])
            isotherlist.extend(listed_data_cur['isother'])
            entitylist.extend(listed_data_cur['labels'])
            iscleanlist.extend(listed_data_cur['isclean'])
        
        allisotherlist.extend(isotherlist)
        allentitylist.extend(entitylist)
        allscorelist.extend(scorelist) 
        alliscleanlist.extend(iscleanlist)  
        batch_id+=1    

    with open(config.res_folder+'/confscore_cut.txt', 'w') as f:
        for i in range(len(allscorelist)):     
            f.write("%d %s %.2f %d" % (allisotherlist[i], allentitylist[i], allscorelist[i], alliscleanlist[i]))  
            f.write("\n")
    
    scoresdata = pd.read_csv(config.res_folder+'/confscore_cut.txt', sep=" ", header=None, quoting=csv.QUOTE_NONE)
    scoresdata.columns = ["isother","entity","score","isclean"]

    if config.cutoff=='fake':
        scoresdata['isfake'] = [e.startswith('fake') for e in scoresdata['entity']]
        print(np.sum(scoresdata['isfake']))
    
        posfake = scoresdata[scoresdata['isother']==0]
        posfake = posfake[posfake['isfake']==1]
        poscutoff = np.quantile(posfake['score'], q=config.fakeq)
        
        negfake = scoresdata[scoresdata['isother']==1]
        negfake = negfake[negfake['isfake']==1]
        negcutoff = np.quantile(negfake['score'], q=config.fakeq)

        config.poscutoff = poscutoff
        if config.dataset.startswith('massive_en_us__noise_miss') or config.dataset.startswith('conll_noise_miss') or config.dataset in ['conll_transfer', 'conll_dist', 'massive_transeasy', 'massive_dist','wikigold_dist','wikigold_self']:
            config.negcutoff = negcutoff
        else:
            config.negcutoff = 10000 # if we know there is cannot be any noise happen on non-entity, we do not throw out anything
    
    if config.cutoff=='goracle':
        negscore = np.array(scoresdata[scoresdata['isother']==1]['score'])   
        negres = np.array(scoresdata[scoresdata['isother']==1]['isclean'])
        if len(list(set(negres)))>1:     
            neg_logis_model = LogisticRegression(solver='liblinear', random_state=0)
            neg_logis_model.fit(negscore.reshape(-1,1), negres)
            negpredict = neg_logis_model.predict(negscore.reshape(-1,1))
        else:
            negpredict = negres
        neg_remove_rate = 1- np.mean(negpredict)
        config.negcutoff = np.quantile(negscore, 1-neg_remove_rate)

        posscore = np.array(scoresdata[scoresdata['isother']==0]['score'])  
        posres = np.array(scoresdata[scoresdata['isother']==0]['isclean'])
        if len(list(set(posres)))>1:     
            pos_logis_model = LogisticRegression(solver='liblinear', random_state=0)
            pos_logis_model.fit(posscore.reshape(-1,1), posres)
            pospredict = pos_logis_model.predict(posscore.reshape(-1,1))
        else:
            pospredict = posres

        pos_remove_rate = 1-np.mean(pospredict)
        config.poscutoff = np.quantile(posscore, 1-pos_remove_rate)

    if config.cutoff=='fitmix':
        negscore = np.array(scoresdata[scoresdata['isother']==1]['score'])
        if config.usecleanscore=='nerloss':
            thred = 0.1
        else:
            thred = 0    
        negcutoff = get_cutoff(negscore, thred = thred, recall=config.recall, usef1=config.usef1)
        if not (config.dataset.startswith('massive_en_us__noise_miss') or config.dataset.startswith('conll_noise_miss') or config.dataset in ['conll_transfer', 'conll_dist', 'massive_transeasy', 'massive_dist','wikigold_dist','wikigold_self']):
            negcutoff = 10000
        config.negcutoff = negcutoff
        
        posscore = np.array(scoresdata[scoresdata['isother']==0]['score'])
        poscutoff = get_cutoff(posscore, thred = 0, recall=config.recall, usef1=config.usef1)
        config.poscutoff = poscutoff
    
    os.remove(config.res_folder+'/confscore_cut.txt')
    

# compute and visualize all the kinds of confidence score computed from our clean anchor model
def evaluate_cleanmodel(config: Config, batched_data, list_data):

    alliscleanlist = []
    allisotherlist = []
    allscorelist = []
    allwordslist = []
    allentitylist = []
    allgoldentitylist = []
    allscorechoiceslist = {}
    
    batch_id = 0
    for batch in batched_data:
        iscleanlist = []
        isotherlist = []
        scorelist = [] 
        scorechoices_list = {s: [] for s in ['nerloss', 'encoderloss', 'diff', 'entropy', 'spike','aum']}
        entitylist= []
        wordslist = []
        goldentitylist = [] 

        for idx in range(len(batch['word_seq_lens'])):
            list_idx = batch_id * config.batch_size + idx
            listed_data_cur = list_data[list_idx]
            scorelist.extend(listed_data_cur['scores'])
            for otherscore in scorechoices_list.keys():
                if otherscore in listed_data_cur:
                    scorechoices_list[otherscore].extend(listed_data_cur[otherscore])
            # if applying on train set, then also extract the is_noisy label
            is_noise = listed_data_cur['isclean']  
            iscleanlist.extend(is_noise)
            isotherlist.extend(listed_data_cur['isother'])
            goldentitylist.extend(listed_data_cur['gold_labels'])
            entitylist.extend(listed_data_cur['labels'])
            wordslist.extend(listed_data_cur['words'])
        
        allwordslist.extend(wordslist)
        allentitylist.extend(entitylist)
        allgoldentitylist.extend(goldentitylist)
        alliscleanlist.extend(iscleanlist)
        allisotherlist.extend(isotherlist)
        allscorelist.extend(scorelist)   
        for otherscore in scorechoices_list.keys(): 
            if otherscore not in allscorechoiceslist:
                allscorechoiceslist[otherscore] = []
            allscorechoiceslist[otherscore].extend(scorechoices_list[otherscore])  
        batch_id+=1    

    with open(config.res_folder+'/confscore_all.txt', 'w') as f:
        for i in range(len(allscorelist)):     
            f.write("%s %s %s %d %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f" % (allwordslist[i], allentitylist[i], allgoldentitylist[i], alliscleanlist[i], allisotherlist[i], allscorelist[i], \
                    allscorechoiceslist['nerloss'][i], allscorechoiceslist['encoderloss'][i], allscorechoiceslist['diff'][i], allscorechoiceslist['entropy'][i], allscorechoiceslist['spike'][i], allscorechoiceslist['aum'][i]))  
            f.write("\n")
    
    scoresdata = pd.read_csv(config.res_folder+'/confscore_all.txt', sep=" ", header=None, quoting=csv.QUOTE_NONE)
    scoresdata.columns = ["words", "entity", "goldentity", "isclean", "isother", "score", "nerloss", "encoderloss", "diff", "entropy", "spike","aum"]
    if config.cutoff=='fake':
        scoresdata['isfake'] = [e.startswith('fake') for e in scoresdata['entity']]
        print(np.sum(scoresdata['isfake']))
    for name in ["nerloss", "encoderloss", "diff", "entropy", "spike","aum"]:     
        if name=='spike':
            scoresdata['spike'] = 1-scoresdata['spike']
        plt.figure(figsize=(5,4))
        sns.histplot(data=scoresdata[scoresdata['entity']=='O'], x=name, hue="isclean", bins = 50, stat = 'density')
        plt.savefig(config.res_folder+"/"+ name +"_neg.png", format="png")     
        plt.figure(figsize=(5,4))
        sns.histplot(data=scoresdata[scoresdata['entity']!='O'], x=name, hue="isclean", bins = 50, stat = 'density')
        plt.savefig(config.res_folder+"/"+ name +"_pos.png", format="png") 
        if config.cutoff=='fake':
            plt.figure(figsize=(5,4))
            sns.histplot(data=scoresdata[scoresdata['isfake']==True], x=name, hue="isother", bins = 50, stat = 'density')
            plt.savefig(config.res_folder+"/"+ name +"_posfake.png", format="png") 
        scoredatapos = scoresdata[scoresdata['entity']!='O']
        # scoredatapos = scoredatapos[scoredatapos['score']>0]
        freq = dict(collections.Counter([g[2:] for g in scoredatapos['goldentity']]))
        freq = {e: v/len(scoredatapos) for e,v in freq.items()}
        listval = list(freq.values())
        listkeys = list(freq.keys())
        idx = np.argsort(listval)
        cum = np.cumsum([listval[i] for i in idx])
        key = [listkeys[i] for i in idx]
        cumfreq = {key[i]: cum[i] for i in range(len(idx))}
        quant = [0*(cumfreq[e[2:]]<0.2) + 1*(cumfreq[e[2:]]>=0.2 and cumfreq[e[2:]]<0.4) + 2*(cumfreq[e[2:]]>=0.4 and cumfreq[e[2:]]<0.6) + 3*(cumfreq[e[2:]]>=0.6 and cumfreq[e[2:]]<0.8) + 4*(cumfreq[e[2:]]>=0.8) for e in scoredatapos['goldentity']]
        scoredatapos.loc[:,'quant'] = quant
        plt.figure(figsize=(5,4))
        sns.histplot(data=scoredatapos[scoredatapos['isclean']==0], x=name, hue="quant", bins = 50, stat = 'density')
        plt.savefig(config.res_folder+"/"+ name +"_noise.png", format="png")  
        # scoredatapos = scoredatapos[scoredatapos[name]>0]
        plt.figure(figsize=(5,4))
        sns.histplot(data=scoredatapos[scoredatapos['isclean']==1], x=name, hue="quant", bins = 50, stat = 'density')
        plt.savefig(config.res_folder+"/"+ name +"_clean.png", format="png")  






def attach_cross_score(config: Config, model, 
                            fold_batches: List[dict],
                            folded_insts: MapStyleJsonDataset,
                            folded_ids: List[int]):
    """
    assign the confidence scores estimated by out-of-sample error to the training data
    """
    batch_id = 0
    batch_size = config.batch_size

    model.eval()
    for batch in fold_batches:
        encoder_scores,_= model.encoder(batch['input_ids'], batch['word_seq_lens'], batch['attention_mask'], batch['token2word_mask'])
        onehot_label = torch.zeros_like(encoder_scores).scatter_(-1, batch['labels_id'].unsqueeze(-1), 1)
        maxscore = torch.max(encoder_scores, dim=-1).values
        obsscore = (encoder_scores * onehot_label).sum(dim=-1)
        confscore = (maxscore-obsscore).detach().cpu().numpy() 
        word_seq_lens = batch['word_seq_lens'].cpu().numpy()
        for idx in range(len(word_seq_lens)):
            length = word_seq_lens[idx]
            fold_inst_idx = batch_id * batch_size + idx
            inst_idx = folded_ids[fold_inst_idx]
            folded_insts.update(inst_idx, 'scores', confscore[idx,:length])
        batch_id += 1


#============ the following are only used in CLout method


def hard_constraint_predict(config: Config, model, 
                            fold_batches: List[dict],
                            folded_insts: MapStyleJsonDataset,
                            folded_ids = List[str]):
    """
    assign the observed labels as predicted labels to the training data
    """
    batch_id = 0
    batch_size = config.batch_size
    model.eval()
    for batch in fold_batches:
        #one_batch_insts = folded_insts[batch_id * batch_size:(batch_id + 1) * batch_size]
        batch_max_scores, batch_max_ids = model.decode(batch)
        batch_max_ids = batch_max_ids.cpu().numpy()
        word_seq_lens = batch['word_seq_lens'].cpu().numpy()
        for idx in range(len(batch_max_ids)):
            length = word_seq_lens[idx]
            prediction = batch_max_ids[idx][:length].tolist()
            prediction = prediction[::-1] # this is the right way!!
            fold_inst_idx = batch_id * batch_size + idx
            inst_idx = folded_ids[fold_inst_idx]
            folded_insts.update(inst_idx, 'labels_id', prediction)
        batch_id += 1


def ranknoisy(config: Config, model, fold_batches: List[dict],
                            folded_insts: MapStyleJsonDataset,
                            folded_ids = List[str]):
    """
    assign the confidence score to the training data, and rank it accordingly
    """
    batch_id = 0
    batch_size = config.batch_size
    model.eval()
    scores = []
    for batch in fold_batches:
        batch_max_scores, _ = model.decode(batch)
        batch_max_scores = batch_max_scores.detach().cpu().numpy().reshape(1,-1).tolist()
        scores.extend(flatten(batch_max_scores)) # this score is utterance level
        batch_id += 1
    return scores


def weightnoisy(config: Config, model, fold_batches: List[dict],
                            folded_insts: MapStyleJsonDataset,
                            folded_ids = List[str]):
    """
    assign the weights to the training data
    """
    batch_id = 0
    batch_size = config.batch_size
    model.eval()
    for batch in fold_batches:
        #words, word_seq_lens, batch_context_emb, chars, char_seq_lens,_,_,_,_,_,_,_,_= batch
        #one_batch_insts = folded_insts[batch_id * batch_size:(batch_id + 1) * batch_size]
        encoder_scores,_ = model.encoder(batch['input_ids'], batch['word_seq_lens'], batch['attention_mask'], batch['token2word_mask'])
        marginals = model.inferencer.marginal(encoder_scores, batch['word_seq_lens'])
        marginals = marginals.detach().cpu().numpy()
        word_seq_lens = batch['word_seq_lens'].cpu().numpy()
        
        for idx in range(len(marginals)):
            length = word_seq_lens[idx]
            fold_inst_idx = batch_id * batch_size + idx
            inst_idx = folded_ids[fold_inst_idx]
            folded_insts.update(inst_idx,'weights',marginals[idx, :length, :])
        batch_id += 1




