from processer.noise import *

#====== get bias pairs
words_list, gold_labels_list, entity_words, gold_entity_labels  = list_colln('../data/conll/train.txt', otag='O')
biaspairs, ambwords  = getbiaspair(entity_words, gold_entity_labels)
bias = getbiask(biaspairs, 4)   

# use the top 10 ambigous slots 
for b in bias.items():    
   print(b)

for noisetype in ['miss','over','shift','shrink','extend','swap','bias']:
    for noiselevel in [1, 0.7, 0.5, 0.3]:
        gen_noisy_data_colln(words_list, gold_labels_list, method = noisetype, noiselevel = noiselevel, 
                        filepath = '../data/conll',
                        bias = bias)


#====== train-dev split of wikidata
words_list, gold_labels_list, entity_words, gold_entity_labels  = list_colln('../data/wikigold/train.txt', otag='O')
# 1841 
indices = list(range(len(words_list)))
random.shuffle(indices)
train_indices = indices[:int(0.7*len(words_list))]
dev_indices = indices[int(0.7*len(words_list)):]

with open('../data/wikigold'+'/newtrain.txt', 'w') as f:
    for i in train_indices:
        gold_labels = [g[2:] if g!='O' else g for g in gold_labels_list[i]]
        gold_labels = convert_tag_to_BIO(gold_labels)
        for j in range(len(words_list[i])):     
            f.write("%s %s \n" % (words_list[i][j], gold_labels[j]))
        f.write("\n")   

with open('../data/wikigold'+'/newdev.txt', 'w') as f:
    for i in dev_indices:
        gold_labels = [g[2:] if g!='O' else g for g in gold_labels_list[i]]
        gold_labels = convert_tag_to_BIO(gold_labels)
        for j in range(len(words_list[i])):     
            f.write("%s %s \n" % (words_list[i][j], gold_labels[j]))
        f.write("\n")

with open('../data/wikigold'+'/newtest.txt', 'w') as f:
    for i in dev_indices:
        gold_labels = [g[2:] if g!='O' else g for g in gold_labels_list[i]]
        gold_labels = convert_tag_to_BIO(gold_labels)
        for j in range(len(words_list[i])):     
            f.write("%s %s \n" % (words_list[i][j], gold_labels[j]))
        f.write("\n")

import pickle
f = open('./results/wikigold/bislstmfinal1more_____none/results/test_pred.pkl', 'rb')
test_pred = pickle.load(f)
f.close()

with open('../data/conll/transtrain.txt', 'w') as f:
    for i in test_pred.ids:
        pred_labels = convert_tag_to_BIO([p[2:] if p!='O' else p for p in test_pred.updated[i]['prediction']])
        gold_labels = convert_tag_to_BIO([g[2:] if g!='O' else g for g in test_pred[i]['gold_labels']])
        words = test_pred[i]['words']
        for j in range(len(words)):     
            f.write("%s %s %s\n" % (words[j], pred_labels[j], gold_labels[j]))
        f.write("\n")

 

