import json as json
import collections
import random
import os
import matplotlib.pyplot as plt
import functools
from random import sample
import itertools
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import shutil
from processer.tokenization import convert_tag_to_BIO


#===== compute the transition matrix among entity classes
def getbiaspair(entity_words, gold_entity_labels):
    '''
    param
        entity_words: list of all entity words
        gold_entity_labels: list of all gold entity labels 
    return
        biaspairs: a dictionary of all the entity pairs that has shared words, 
                  where the value is the number of shared words they have
        ambwords: a dictionary of all the words that has different entities, 
                  where the value is the number of entity types that the word has been tagged with
    '''
    err = 0
    biaspairs = {} # hard entity
    ambwords = {} # hard words
    allcontent = set(entity_words)
    for content in allcontent:
        id = np.where(np.array(entity_words)==content)[0]
        slot = np.array(gold_entity_labels)[id]
        repslot = list(collections.Counter(slot))
        if len(repslot)>1:
            maxcount = max(dict(collections.Counter(slot)).values())
            disagree = len(slot)-maxcount
            if content not in ambwords:
                ambwords[content] = disagree
            else:
                ambwords[content] += disagree
            
            for pairs in itertools.combinations(repslot, 2):
                if tuple(sorted(pairs)) in biaspairs:
                    biaspairs[tuple(sorted(pairs))] +=1
                else:
                    biaspairs[tuple(sorted(pairs))] = 1        
            err+=1
    print('confusion proportion out of all entity words: %.3f'%(err/len(allcontent)))
    print('number of entity pairs that has overlapping words: %d'%(len(biaspairs)))   

    return biaspairs, ambwords       

def getbiask(biaspairs, topk):
    '''
    param
        biaspairs: the biaspairs computed from getbiaspair function
        topk: an integer indicates how many bias pairs one would like to consider
    return
        bias: a dictionary presenting the trasition matrix of the top k entity pairs that has the most number of shared words, 
              where the key is one entity class A, and the value is a dictionary containing {entity class B: the number of shared words with it; ...}.
    '''
    biaspairs_sorted = {k: v for k, v in sorted(biaspairs.items(), key=lambda item: -item[1])}
    bias = {}
    count = 0
    for pair, v in biaspairs_sorted.items():
        if count < topk:
            print(pair, v)
            if pair[0] not in bias:
                bias[pair[0]] = {pair[1]: v}
            else:
                bias[pair[0]][pair[1]] = v      
            if pair[1] not in bias:
                bias[pair[1]] = {pair[0]: v}
            else:
                bias[pair[1]][pair[0]] = v    
            count+=1    
        else:
            break
    return bias            

def flatten(t):
    """
    flatten a list of list into a list
    """
    return functools.reduce(lambda x,y: x+y,t)

def tokenize(word):
    ans = []
    ans.append(word)
    # to do
    return ans 

#===== add noise to a list of tags
def noisytag(taglist, level = 0.2, method = 'None', allslots=None, forbid = None, bias = None, otag = 'O'):
    '''
    param
        taglist: a list of tags, e.g. ['Other','person','Other','location','location','Other']
        level: proportion of entities to add noise
        method: noise type
        allslots: a list containing all possible entity types
        forbid: a list of entities, which are forbidden to add noise
        bias: a dictionary {entity-a: {entity-b: 2; entity_c: 3}, {entity-c: {entity-b: 4, entity-d: 5}}}, 
            indicating the transition matrix between easy to confused entity pairs
        otag: what is used to represent 'Other'    
    return
        newtaglist: the list of tags after perturbation      
    '''
    newtaglist = taglist.copy()
    if method=='mixed':
        methodslist = ['miss','over','shift','extend','shrink','swap','bias']
        choose_id = sample(range(len(methodslist)),1)[0]
        method = methodslist[choose_id] 
    if method =='shift':
        # find the start and end of an entity span (len >1), and shit it one word to the left or right    
        idx = []
        idxend = []
        i=1
        while i< len(taglist)-1:
            if taglist[i]!=otag and taglist[i-1]==otag and taglist[i]==taglist[i+1]:
                j=i+1
                while j< len(taglist)-1 and taglist[j]==taglist[i]:
                    j+=1
                if taglist[j]==otag:   
                    if (forbid is None) or ((forbid is not None) and (taglist[j-1] not in forbid)): 
                      idx.append(i)
                      idxend.append(j-1) 
                i=j
            else:
                i+=1                               
        if len(idx)>0:
            selidx = random.sample(range(len(idx)), max(1, int(level*len(idx))))
            for k in selidx:
                if np.random.binomial(1, 0.5)==1:
                    newtaglist[idxend[k]+1] = taglist[idx[k]]
                    newtaglist[idx[k]] = otag       
                else:
                    newtaglist[idxend[k]] = otag
                    newtaglist[idx[k]-1] = taglist[idx[k]]                
    if method =='over':
        # randomly change a non-entity spot to entity    
        idx = list(np.where(np.array(taglist)==otag)[0])
        if len(idx)>0:
            selidx = random.sample(idx, max(1, int(level*len(idx))))
            for i in selidx:
                newtaglist[i] = random.sample(allslots, 1)[0]    
    if method == 'miss':
        # randomly change a entity spot to nonentity
        if forbid is not None:
            forb = [otag]+forbid 
            idx = list(np.where(np.array([(t not in forb) for t in taglist]))[0])
        else:  
            idx = list(np.where(np.array(taglist)!=otag)[0])
        if len(idx)>0:
            selidx = random.sample(idx, max(1,int(level*len(idx))))
            for i in selidx:
                newtaglist[i] = otag
    if method == 'extend':
        # randomly extend an entity span by one word 
        idx = []
        idxend = []
        i=0
        while i<len(taglist)-1:
            if taglist[i]!=otag and taglist[i-1]!=taglist[i]:
                j=i+1
                while j<len(taglist) and taglist[j]==taglist[i]:
                    j+=1 
                if (forbid is None) or ((forbid is not None) and (taglist[j-1] not in forbid)):       
                  idx.append(i)
                  idxend.append(j-1) 
                i=j
            else:
                i+=1
        if len(idx)>0:
            selidx = random.sample(range(len(idx)), max(1,int(level*len(idx))))
            for i in selidx:
                curtag = newtaglist[idx[i]]
                if idx[i]==0 and idxend[i]< len(newtaglist)-1 and newtaglist[idxend[i]+1]==otag:
                    newtaglist[idxend[i]+1] = curtag
                if idx[i]>0 and idxend[i]== len(newtaglist)-1 and newtaglist[idx[i]-1]==otag:
                    newtaglist[idx[i]-1] = curtag    
                if idx[i]>0 and idxend[i]<len(newtaglist)-1:
                    if np.random.binomial(1, 0.5)==1:
                        newtaglist[idx[i]-1] = curtag
                    else:
                        newtaglist[idxend[i]+1] = curtag             
    if method == 'shrink':
        # randomly shrink an entity span (len >1) by one word 
        idx = []
        idxend = []
        i=0
        while i< len(taglist)-1:
            if taglist[i]!=otag and taglist[i-1]!=taglist[i] and taglist[i]==taglist[i+1]:
                j=i+1
                while j<len(taglist) and taglist[j]==taglist[i]:
                    j+=1
                if (forbid is None) or ((forbid is not None) and (taglist[j-1] not in forbid)):           
                    idx.append(i)
                    idxend.append(j-1) 
                i=j
            else:
                i+=1
        if len(idx)>0:
            selidx = random.sample(range(len(idx)), max(1,int(level*len(idx))))
            for i in selidx:
                if np.random.binomial(1, 0.5)==1:
                    newtaglist[idx[i]] = otag
                else:
                    newtaglist[idxend[i]] = otag                              
    if method == 'swap':
        # randomly change an entity span to another entity tag
        idx = []
        idxend = []
        i=0
        while i< len(taglist):
            if taglist[i]!=otag and taglist[i-1]!=taglist[i]:
                j=i+1
                while j<len(taglist) and taglist[j]==taglist[i]:
                    j+=1 
                if (forbid is None) or ((forbid is not None) and (taglist[j-1] not in forbid)):            
                    idx.append(i)
                    idxend.append(j-1) 
                i=j
            else:
                i+=1        
        if len(idx)>0:
            selidx = random.sample(range(len(idx)), max(1,int(level*len(idx))))
            for i in selidx:
                curtag = newtaglist[idx[i]]
                newtag = random.sample(allslots, 1)[0]
                if newtag!=curtag:
                    for k in range(idx[i],idxend[i]+1):
                        newtaglist[k] =newtag    
    if method == 'bias':
        # randomly change an entity span to some specific entity tags, using the pattern in the data itself
        idx = []
        idxend = []
        i=0
        while i< len(taglist):
            if (taglist[i]!=otag) and (taglist[i-1]!=taglist[i]) and (taglist[i] in bias):
                j=i+1
                while j<len(taglist) and taglist[j]==taglist[i]:
                    j+=1
                if (forbid is None) or ((forbid is not None) and (taglist[j-1] not in forbid)):       
                    idx.append(i)
                    idxend.append(j-1) 
                i=j
            else:
                i+=1
        if len(idx)>0:
            selidx = random.sample(range(len(idx)), max(1,int(level*len(idx))))
            for i in selidx:
                curtag = newtaglist[idx[i]]
                options = list(bias[curtag].keys())
                weights = [v for k,v in bias[curtag].items()]
                weights = weights/np.sum(weights)
                newtag = random.choices(population = options, k=1, weights=weights)[0]
                for k in range(idx[i],idxend[i]+1):
                    newtaglist[k] =newtag  
    return newtaglist  


#======== for massive format data
def list_massive(filename, otag='O'):
    """
    read from data of massive format
    param
        filename: the name of the massive jsonl file to read
        otag: what is used to represent the 'Other' tag
    return
        words_list: a list of list containing utternaces of words
        gold_labels_list: a list of list containing utternaces of words
        entity_words: a list containing all the entity words
        gold_entity_labels: a list containing all the gold entity labels  
        domain_list: a list of all utterance domains
        intent_list: a list of all utternace intents
        domains: a list containing the domains for all the entity words
        intents: a list containing the intents for all the entity words
    """
    
    with open(filename,'r') as json_file:
        json_list = list(json_file)

    annot_list = []
    domain_list = []
    intent_list = []
    for json_str in json_list:
        result = json.loads(json_str)
        annot_list.append(result['annot_utt'])
        domain_list.append(result['scenario'])
        intent_list.append(result['intent'])
    
    num_insts = len(annot_list)
    words_list = []
    gold_labels_list = []
    
    entity_words = []
    gold_entity_labels = []
    domains = []
    intents = []

    for u in range(num_insts):
        utt = annot_list[u]
        utt = utt.replace('[','[ ')
        utt = utt.replace(']',' ]')
        utt_list = utt.split(' ')
        i = 0
        words = []
        gold_labels = []
        while i < len(utt_list):
            if utt_list[i] =="[":
                tagnow = utt_list[i+1]
                j = i+3 # skip ':'
                while utt_list[j]!=']':
                    for w in tokenize(utt_list[j]):
                        words.append(w)
                        gold_labels.append(tagnow)
                        entity_words.append(w)
                        gold_entity_labels.append(tagnow)
                        domains.append(domain_list[u])
                        intents.append(intent_list[u])
                    j+=1
                i=j+1    
            else:
                for w in tokenize(utt_list[i]):
                    words.append(w)
                    gold_labels.append(otag)
                i+=1
        words_list.append(words)
        gold_labels_list.append(gold_labels) 

    return words_list, gold_labels_list, entity_words, gold_entity_labels, domain_list, intent_list, domains, intents



def gen_noisy_data_massive(words_list, gold_labels_list,
                   method = 'None', 
                   train_dev_test = [0.5,0.25,0.25], 
                   noiselevel = 0.5, scoremethod = 'micro',
                   filepath = './data/massive',
                   forbid = None,
                   bias = None,
                   seed = None,
                   otag = 'O'):
    '''
    generate the noisy version of data for massive formated data, where we also need to do train-dev-test split
    param
        word_list: a list of list containing all words for all utterances
        gold_labels_list: a list of list containing the gold labels for all words for all utterances
        method: noise type
        train_dev_test: list of size 3, indicating the proportion of train, dev and test, should add to one
        noise level: the proportion of utterance to add noise
        scoremethod: the method to compute the f1 score of the noisy data set
        filepath: the path to read and save data
        forbid: a list entities that are forbidden to add noise
        bias: a dictionary {entity-a: {entity-b: 2; entity_c: 3}, {entity-c: {entity-b: 4, entity-d: 5}}}, 
            indicating the transition matrix between easy to confused entity pairs
        seed: what random seed to use
        otag: what to use to represent 'Other'   
    return
        no return, directly write into a txt file with the noisy labels        
    '''               
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    indices = list(range(len(words_list)))
    random.shuffle(indices)
    
    train_indices = indices[:int(train_dev_test[0]*len(words_list))]
    test_indices = indices[int(train_dev_test[0]*len(words_list)):int((train_dev_test[0]+train_dev_test[1])*len(words_list))]
    dev_indices = indices[int((train_dev_test[0]+train_dev_test[1])*len(words_list)):]
    allsubsets = [train_indices, test_indices, dev_indices]
    allnames = ['train','test','dev']

    allslots = list(set(flatten(gold_labels_list)))

    # get noisy data
    for s in range(len(allnames)):
        fullpath = filepath + '_noise_'+ method +'_level'+ str(noiselevel)
        if not os.path.exists(fullpath):
            os.makedirs(fullpath)   
            labels_list = []
            with open(fullpath + '/' + allnames[s] + '.txt', 'w') as f:
                for i in allsubsets[s]:
                    labels = []
                    gold_labels = gold_labels_list[i]
                    if np.random.binomial(1, noiselevel)==1:
                        for j in range(len(words_list[i])):
                            labels.append(gold_labels[j])
                        labels = noisytag(labels, forbid = forbid, bias=bias, seed=seed, method=method, allslots = allslots, otag=otag)  
                        labels = convert_tag_to_BIO(labels)
                        gold_labels = convert_tag_to_BIO(gold_labels)
                    else:
                        gold_labels = convert_tag_to_BIO(gold_labels)
                        labels = gold_labels
                    labels_list.append(labels)               
                    for j in range(len(words_list[i])):     
                        f.write("%s %s %s \n" % (words_list[i][j], labels[j], gold_labels[j]))
                    f.write("\n")    
        if allnames[s]=='train':
            precision = precision_score(flatten(gold_labels_list), flatten(labels_list), average=scoremethod)
            recall = recall_score(flatten(gold_labels_list), flatten(labels_list), average=scoremethod)
            f1 = f1_score(flatten(gold_labels_list), flatten(labels_list), average=scoremethod)
            print('noisy method', method, 'f1 score: ', np.round(f1,3), ' precision', np.round(precision,3),' recall', np.round(recall,3))


#======== for colln format data
def list_colln(filename, otag='O'):
    '''
    read from data of conll format
    param
        filename: the name of the massive jsonl file to read
        otag: what is used to represent the 'Other' tag
    return
        words_list: a list of list containing utternaces of words
        gold_labels_list: a list of list containing utternaces of words
        entity_words: a list containing all the entity words
        gold_entity_labels: a list containing all the gold entity labels 
    '''               
    print("Reading file: " + filename)
    with open(filename, 'r', encoding='utf-8') as f:
        words_list = []
        gold_labels_list = []

        words = []
        gold_labels = []
        
        entity_words = []
        gold_entity_labels = []
        
        for line in f.readlines():
            line = line.rstrip()
            
            if line == "": # reach the instance break

                words_list.append(words)
                gold_labels_list.append(gold_labels)

                words = []
                gold_labels=[]
                continue
            
            if(len(line.split())==1): # one columns                
                gold_label=line.split()[0]
                word=','
            else:
                if(len(line.split())==2): # two columns
                    word, gold_label = line.split()[0], line.split()[1]
                    
            words.append(word)
            gold_labels.append(gold_label) 
            if gold_label!=otag:
                entity_words.append(word)
                gold_entity_labels.append(gold_label[2:])

    f.close()

    return words_list, gold_labels_list, entity_words, gold_entity_labels  


def gen_noisy_data_colln(words_list, gold_labels_list, 
                   method = 'None', 
                   noiselevel = 0.5, scoremethod = 'micro',
                   filepath = './data/colln',
                   forbid = None,
                   bias = None,
                   seed = None,
                   otag='O'):
    '''
    generate the noisy version of data for conll formated data, where trainining samples are fixed
    param
        word_list: a list of list containing all words for all utterances
        gold_labels_list: a list of list containing the gold labels for all words for all utterances
        method: noise type
        noise level: the proportion of utterance to add noise
        scoremethod: the method to compute the f1 score of the noisy data set
        filepath: the path to read and save data
        forbid: a list entities that are forbidden to add noise
        bias: a dictionary {entity-a: {entity-b: 2; entity_c: 3}, {entity-c: {entity-b: 4, entity-d: 5}}}, 
            indicating the transition matrix between easy to confused entity pairs
        seed: what random seed to use
        otag: what to use to represent 'Other'   
    return
        no return, directly write into a txt file with the noisy labels 
    '''   

    # get all unique entities
    allslots = list(set(flatten(gold_labels_list)))
 
    fullpath = filepath + '_noise_'+ method +'_level'+ str(noiselevel)
    if not os.path.exists(fullpath):
        os.makedirs(fullpath)   
    
    labels_list = []
    with open(fullpath + '/train.txt', 'w') as f:
        
        for i in range(len(words_list)):
            labels = []
            gold_labels = gold_labels_list[i]
            if np.random.binomial(1, noiselevel)==1:
                for j in range(len(words_list[i])):
                    if gold_labels[j]==otag:
                        labels.append(gold_labels[j])
                    else:
                        labels.append(gold_labels[j][2:]) 
                labels = noisytag(labels, forbid = forbid, bias=bias, seed=seed, method=method, allslots = allslots, otag=otag)  
                labels = convert_tag_to_BIO(labels) 
            else:
                labels = gold_labels
            labels_list.append(labels)

            for j in range(len(words_list[i])):     
                f.write("%s %s %s \n" % (words_list[i][j], labels[j], gold_labels_list[i][j]))
            f.write("\n")    

    shutil.copyfile(filepath+'/test.txt', fullpath + '/test.txt')
    shutil.copyfile(filepath+'/dev.txt', fullpath + '/dev.txt')

    precision = precision_score(flatten(gold_labels_list), flatten(labels_list), average=scoremethod)
    recall = recall_score(flatten(gold_labels_list), flatten(labels_list), average=scoremethod)
    f1 = f1_score(flatten(gold_labels_list), flatten(labels_list), average=scoremethod)
    print('noisy method', method, 'f1 score: ', np.round(f1,3), ' precision', np.round(precision,3),' recall', np.round(recall,3))

 