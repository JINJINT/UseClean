from calendar import calendar

from numpy import intersect1d
from processer.noise import *

words_list, gold_labels_list, entity_words, gold_entity_labels, domain_list, intent_list, domains, intents  = list_massive('../data/massive/en-US.jsonl', otag='O')
biaspairs, ambwords  = getbiaspair(entity_words, gold_entity_labels)
bias = getbiask(biaspairs, 4)   

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
allent = {g:i for i,g in enumerate(set((gold_entity_labels)))}

d = {'words': entity_words, 'domains': domains, 'intents': intents, 'entity': gold_entity_labels }
dom_ent = {}
df = pd.DataFrame(data=d)
overlap = np.zeros((18,18))
for i, domain1 in enumerate(set(domains)):
    subdata1 = df[df['domains']==domain1]
    dom_ent[domain1] = subdata1['entity']
    for j, domain2 in enumerate(set(domains)):     
        subdata2 = df[df['domains']==domain2]
        over = set.intersection(set(subdata1['entity']), set(subdata2['entity']))
        overlap[i,j] = len(over)/len(set(subdata1['entity']))
        overlap[j,i] = len(over)/len(set(subdata2['entity']))

np.fill_diagonal(overlap, 1)

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
import numpy as np


Z = hierarchy.ward(overlap)
a = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z, overlap))

np.fill_diagonal(overlap, 1)
overlap_order = overlap[list(a), :]
overlap_order = overlap_order[:,list(a)]
f, ax = plt.subplots(figsize=(9, 7))
ax = sns.heatmap(overlap_order, cmap="YlGnBu", xticklabels=set(domains), yticklabels=set(domains))
plt.savefig('/Users/jinjint/Desktop/massivedoaminratio.png')
        

list1 = ['takeaway', 'news', 'recommendation', 'transport','general','weather','iot','qa']
list2 = ['cooking', 'music', 'play','audio','lists','social','calendar','email','alarm','datetime']  



dom_ent_len = {d:len(v) for d,v in dom_ent.items()}
{'takeaway': 721, 'news': 1236, 'recommendation': 1002, 'transport': 1714, 'alarm': 915, 'datetime': 959, 'lists': 554, 'weather': 1612, 'iot': 1269, 
 'email': 1703, 'qa': 2085, 'audio': 173, 'general': 434, 'calendar': 5927, 'social': 910, 'cooking': 450, 'music': 232, 'play': 3698}

dom_ent_ulen = {d:len(set(v)) for d,v in dom_ent.items()}
{'takeaway': 14, 'news': 10, 'recommendation': 14, 'transport': 16, 'alarm': 13, 'datetime': 8, 'lists': 17, 'weather': 10, 'iot': 13, 
'email': 17, 'qa': 15, 'audio': 8, 'general': 18, 'calendar': 20, 'social': 14, 'cooking': 14, 'music': 15, 'play': 26}

len1 = sum([dom_ent_len[d] for d in list1]) # 10073
len2 = sum([dom_ent_len[d] for d in list2]) #15521

set1 = set(flatten([list(dom_ent[d]) for d in list1]))
set2 = set(flatten([list(dom_ent[d]) for d in list2]))
len(set1)
len(set2)

inst1 = [i for i, domain in enumerate(domain_list) if domain in list1] # 7078
inst2 = [i for i, domain in enumerate(domain_list) if domain in list2] # 9443

indices = inst1
random.shuffle(indices)
train_indices = indices[:int(0.7*len(inst1))]
dev_indices = indices[int(0.7*len(inst1)):]

with open('../data/massive_sub/train.txt', 'w') as f:
    for i in train_indices:
        gold_labels = [g for g in gold_labels_list[i]]
        gold_labels = convert_tag_to_BIO(gold_labels)
        for j in range(len(words_list[i])):     
            f.write("%s %s \n" % (words_list[i][j], gold_labels[j]))
        f.write("\n")  

with open('../data/massive_sub/dev.txt', 'w') as f:
    for i in dev_indices:
        gold_labels = [g for g in gold_labels_list[i]]
        gold_labels = convert_tag_to_BIO(gold_labels)
        for j in range(len(words_list[i])):     
            f.write("%s %s \n" % (words_list[i][j], gold_labels[j]))
        f.write("\n")  

with open('../data/massive_sub/test.txt', 'w') as f:
    for i in inst2:
        gold_labels = [g for g in gold_labels_list[i]]
        gold_labels = convert_tag_to_BIO(gold_labels)
        for j in range(len(words_list[i])):     
            f.write("%s %s \n" % (words_list[i][j], gold_labels[j]))
        f.write("\n")  


import pickle
f = open('./results/massive_sub/bertfinal1_____none/results/test_pred.pkl', 'rb')
test_pred = pickle.load(f)
f.close()

with open('../data/massive/transtrain.txt', 'w') as f:
    for i in test_pred.ids:
        pred_labels = convert_tag_to_BIO([p[2:] if p!='O' else p for p in test_pred.updated[i]['prediction']])
        gold_labels = convert_tag_to_BIO([g[2:] if g!='O' else g for g in test_pred[i]['gold_labels']])
        words = test_pred[i]['words']
        for j in range(len(words)):     
            f.write("%s %s %s\n" % (words[j], pred_labels[j], gold_labels[j]))
        f.write("\n")

#=== easier transfer

list1 = ['takeaway', 'music', 'news', 'social', 'recommendation', 'general','weather','qa','datetime']
list2 = ['cooking', 'play','audio','lists','calendar','email','alarm','iot','transport']  

dom_ent_len = {d:len(v) for d,v in dom_ent.items()}
{'takeaway': 721, 'news': 1236, 'recommendation': 1002, 'transport': 1714, 'alarm': 915, 'datetime': 959, 'lists': 554, 'weather': 1612, 'iot': 1269, 
 'email': 1703, 'qa': 2085, 'audio': 173, 'general': 434, 'calendar': 5927, 'social': 910, 'cooking': 450, 'music': 232, 'play': 3698}

dom_ent_ulen = {d:len(set(v)) for d,v in dom_ent.items()}
{'takeaway': 14, 'news': 10, 'recommendation': 14, 'transport': 16, 'alarm': 13, 'datetime': 8, 'lists': 17, 'weather': 10, 'iot': 13, 
'email': 17, 'qa': 15, 'audio': 8, 'general': 18, 'calendar': 20, 'social': 14, 'cooking': 14, 'music': 15, 'play': 26}

len1 = sum([dom_ent_len[d] for d in list1]) # 10073
len2 = sum([dom_ent_len[d] for d in list2]) #15521

set1 = set(flatten([list(dom_ent[d]) for d in list1]))
set2 = set(flatten([list(dom_ent[d]) for d in list2]))
len(set1) # 36
len(set2) # 50

inst1 = [i for i, domain in enumerate(domain_list) if domain in list1] # 6778
inst2 = [i for i, domain in enumerate(domain_list) if domain in list2] # 9743

indices = inst1
random.shuffle(indices)
train_indices = indices[:int(0.7*len(inst1))]
dev_indices = indices[int(0.7*len(inst1)):]

with open('../data/massive_subeasy/train.txt', 'w') as f:
    for i in train_indices:
        gold_labels = [g for g in gold_labels_list[i]]
        gold_labels = convert_tag_to_BIO(gold_labels)
        for j in range(len(words_list[i])):     
            f.write("%s %s \n" % (words_list[i][j], gold_labels[j]))
        f.write("\n")  

with open('../data/massive_subeasy/dev.txt', 'w') as f:
    for i in dev_indices:
        gold_labels = [g for g in gold_labels_list[i]]
        gold_labels = convert_tag_to_BIO(gold_labels)
        for j in range(len(words_list[i])):     
            f.write("%s %s \n" % (words_list[i][j], gold_labels[j]))
        f.write("\n")  

with open('../data/massive_subeasy/test.txt', 'w') as f:
    for i in inst2:
        gold_labels = [g for g in gold_labels_list[i]]
        gold_labels = convert_tag_to_BIO(gold_labels)
        for j in range(len(words_list[i])):     
            f.write("%s %s \n" % (words_list[i][j], gold_labels[j]))
        f.write("\n")  


import pickle
f = open('./results/massive_sub/bertfinal1_____none/results/test_pred.pkl', 'rb')
test_pred = pickle.load(f)
f.close()

indices = test_pred.ids.copy()
random.shuffle(indices)
train_indices = indices[:int(0.7*len(indices))]
dev_indices = indices[int(0.7*len(indices)):int(0.85*len(indices))]
test_indices = indices[int(0.85*len(indices)):]

with open('../data/massive_trans/train.txt', 'w') as f:    
    for i in train_indices:
        pred_labels = convert_tag_to_BIO([p[2:] if p!='O' else p for p in test_pred.updated[i]['prediction']])
        gold_labels = convert_tag_to_BIO([g[2:] if g!='O' else g for g in test_pred[i]['gold_labels']])
        words = test_pred[i]['words']
        for j in range(len(words)):     
            f.write("%s %s %s\n" % (words[j], pred_labels[j], gold_labels[j]))
        f.write("\n")

with open('../data/massive_trans/dev.txt', 'w') as f:       
    for i in dev_indices:
        gold_labels = convert_tag_to_BIO([g[2:] if g!='O' else g for g in test_pred[i]['gold_labels']])
        words = test_pred[i]['words']
        for j in range(len(words)):     
            f.write("%s %s\n" % (words[j], gold_labels[j]))
        f.write("\n")

with open('../data/massive_trans/test.txt', 'w') as f:       
    for i in test_indices:
        gold_labels = convert_tag_to_BIO([g[2:] if g!='O' else g for g in test_pred[i]['gold_labels']])
        words = test_pred[i]['words']
        for j in range(len(words)):     
            f.write("%s %s\n" % (words[j], gold_labels[j]))
        f.write("\n") 

f = open('./results/massive_subeasy/bertfinal1_____none/results/test_pred.pkl', 'rb')
test_pred = pickle.load(f)
f.close()
indices = test_pred.ids.copy()
random.shuffle(indices)
train_indices = indices[:int(0.7*len(indices))]
dev_indices = indices[int(0.7*len(indices)):int(0.85*len(indices))]
test_indices = indices[int(0.85*len(indices)):]

with open('../data/massive_transeasy/train.txt', 'w') as f:
    for i in train_indices:
        pred_labels = convert_tag_to_BIO([p[2:] if p!='O' else p for p in test_pred.updated[i]['prediction']])
        gold_labels = convert_tag_to_BIO([g[2:] if g!='O' else g for g in test_pred[i]['gold_labels']])
        words = test_pred[i]['words']
        for j in range(len(words)):     
            f.write("%s %s %s\n" % (words[j], pred_labels[j], gold_labels[j]))
        f.write("\n")

with open('../data/massive_transeasy/dev.txt', 'w') as f:       
    for i in dev_indices:
        gold_labels = convert_tag_to_BIO([g[2:] if g!='O' else g for g in test_pred[i]['gold_labels']])
        words = test_pred[i]['words']
        for j in range(len(words)):     
            f.write("%s %s\n" % (words[j], gold_labels[j]))
        f.write("\n")

with open('../data/massive_transeasy/test.txt', 'w') as f:       
    for i in test_indices:
        gold_labels = convert_tag_to_BIO([g[2:] if g!='O' else g for g in test_pred[i]['gold_labels']])
        words = test_pred[i]['words']
        for j in range(len(words)):     
            f.write("%s %s\n" % (words[j], gold_labels[j]))
        f.write("\n") 


