import argparse
from distutils.command.config import config
from email.policy import default
import gc
from trainer.trainer_baseline import *
from trainer.trainer_CLin import *
from trainer.trainer_CLout import *
from trainer.trainer_coreg import *
from trainer.trainer_metaweightnet import *
import pandas as pd
import seaborn as sns
from datastruct.dataset import *
from utils.trans_utils import context_models
import json
from processer.tokenization import _read_txt, build_emb_table, build_word_idx, read_pretrain_embedding

def set_seed(opt, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if opt.device.startswith("cuda"):
        print("using GPU...", torch.cuda.current_device())
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_arguments(parser):

    # dataset parameters
    parser.add_argument('--device', type=str, default="cuda",help="GPU/CPU devices")
    parser.add_argument('--seed', type=int, default=50, help="random seed")
    parser.add_argument('--digit2zero', action="store_true", default=True,
                        help="convert the number to 0, make it true is better")
    parser.add_argument('--dataset', type=str, default="eng_r0.5p0.9")
    parser.add_argument('--embedding_file', type=str, default="../data/glove.txt", # ../data/glove.6B.50d.txt
                        help="we will be using random embeddings if file do not exist")
    parser.add_argument('--embedding_dim', type=int, default=50)
    
    # optimization parameters
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--learning_rate', type=float, default=2e-5) # use 0.01 for bilstm, use 2e-5 fr transformers
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=1e-8)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=10, help="default batch size is 10 (works well)")
    
    # training parameters
    parser.add_argument('--num_epochs', type=int, default=20, help="Usually we set to 20.") # use 30 for bilstm, use 20 for transformers
    parser.add_argument('--train_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--dev_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--test_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--num_outer_iterations', type=int, default=7, help="Number of outer iterations for self-training")

    # encoder hyperparameter
    #===== encoders = {'bilstm','bert-base-uncased', 'bert-base-cased', 'bert-large-cased',
    #                  'openai-gpt','gpt2','ctrl','transfo-xl-wt103','xlnet-base-cased',
    #                  'distilbert-base-cased','roberta-base','roberta-large','xlm-roberta-base'}
    parser.add_argument('--encoder', type=str, default='bert-base-uncased',help='what kind of encoder to use')
    parser.add_argument('--hidden_dim', type=int, default=200, help="hidden size of the encoder") # will be auto adjust to 768 for bert
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout for embedding")
    parser.add_argument('--use_char_rnn', type=int, default=1, choices=[0, 1], help="use character-level lstm, 0 or 1")
    parser.add_argument('--context_emb', type=str, default="none", choices=["elmo", "elmo"],
                        help="contextual word embedding")

    #===== all model parameters
    # cl paper model related
    parser.add_argument('--neg_noise_rate', default=-1,type=float,help="The estimated noise rate of negatives in the first iteration, -1.0 means golden noise rate")
    parser.add_argument('--pos_noise_rate', default=-1,type=float,help="The estimated noise rate of positives in the first iteration, -1.0 means golden noise rate")
    parser.add_argument('--warm_up_num', default=0,type=int)
    parser.add_argument('--num_gradual_neg', default=10,type=int)
    parser.add_argument('--num_gradual_pos', default=10,type=int)
    parser.add_argument('--is_constrain', default=True,type=bool) # whether further calibrate the score
    parser.add_argument('--clmethod', default="CL",type=str) # none, coreg, metaweight, CLin, CLout, CL
    parser.add_argument("--injectclean", default = False, type=bool) # what inject small clean to the training
    parser.add_argument('--modify', default="correct",type=str) # correct, rank, weight (only used in CLout method)
    
    # score related
    parser.add_argument('--score', default="nerloss",type=str) # encoderloss, nerloss, diff, entropy, spike, aum, useclean, usecleanhead, usecleantail
    parser.add_argument('--usescore', default="toss",type=str) # i.e. sample selection
    parser.add_argument("--usecleanscore", default = 'nerloss', type=str) # what confidence score to use in useclean method
    parser.add_argument('--crossfolds', default=2,type=int) # the number of folds in cross-validation (out-sample loss) computation
    parser.add_argument('--cleanprop', default=0.03,type=float) # how many clean data to use: 0.03 for massive, 0.01 for conll, 0.1 for wikigold
    parser.add_argument("--cleanepochs", default = 50, type=int) # how many epoch to train the small clean data, can be higher if cleanprop is smaller

    # cutoff related 
    parser.add_argument('--cutoff', default="heuri",type=str) # heuri, oracle (fit logistic per batch), goracle (fit logistic all samples), clean (only for clpaper), fitmix (only for useclean), fake (only for useclean)
    parser.add_argument("--numfake", default = 10, type=int) # how many more fake examples: #fake = #numfake * #samples_per_class
    parser.add_argument("--fakeq", default = 0.1, type=float) # what quantile to cut fake examples
    parser.add_argument("--usef1", default = True, type=bool) # whether use f1 in fitmix
    parser.add_argument("--recall", default = 0.7, type=float) # the recall lower bound in fitmix

    # semi-supervised learning related
    parser.add_argument("--weight", default = False, type=bool) # whether weight the samples
    parser.add_argument("--warm", default = False, type=bool) # whether warm start with useclean
    
    # contrastive learning related 
    parser.add_argument('--contrastive', default=False,type=bool) # whether add input-contrastive regularization, can only be used if using bert encoder
    parser.add_argument('--lamb', default=0.5,type=float) # the penalty parameter of the input-contrastive regularization
    
    # coreg model related 
    parser.add_argument('--model_nums', default=2,type=int) # number of models in coreg method 
    parser.add_argument("--alpha", type=float, default=3.0) # the penalty parameter of the model-contrastive regularization
    parser.add_argument("--alpha_warmup_ratio", default=0.1, type=float) # the warm up ratio of the contrastive regularization
    
    # backbone related
    parser.add_argument("--classifier", default = 'crf', type=str) # what kind of classifier to use, 'crf' or 'mlp'
    parser.add_argument("--random", default = False, type=bool) # whether use randomized bert
    
    # long tail related
    parser.add_argument("--tau", default = 0, type=float) # whether use logits adjustment

    # saving parameters 
    parser.add_argument('--model_folder', type=str, default="model", help="The name to save the model files")
    parser.add_argument('--res_folder', type=str, default="results", help="The name to save the res files")
    parser.add_argument('--info', type=str, default="alldefault", help="Additional information related to this experiment")
    parser.add_argument('--diag', type=bool, default=False, help="Whether diagonosis this experiment")
    
    args = parser.parse_args()
    
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    
    return args


def main():

    #=== load hyparameter
    parser = argparse.ArgumentParser(description="LSTM CRF implementation")
    opt = parse_arguments(parser) 
    conf = Config(opt)

    #==== get the folder names to save model and results
    if conf.clmethod in ['none', 'coreg','metaweight']:
        conf.num_outer_iterations = 2
        conf.score = ''
        conf.usescore = ''
        conf.cutoff = ''
        conf.modify = ''
    if conf.clmethod=='CLin': 
        conf.num_outer_iterations = 2
        conf.modify = ''
    conf.model_folder = '../results/'+conf.dataset+'/'+conf.info+'_'+conf.score+'_'+conf.usescore+'_'+conf.usecleanscore+'_'+conf.cutoff+'_'+conf.modify+'_'+conf.clmethod+'/'+conf.model_folder
    conf.res_folder = '../results/'+conf.dataset+'/'+conf.info+'_'+conf.score+'_'+conf.usescore+'_'+conf.usecleanscore+'_'+conf.cutoff+'_'+conf.modify+'_'+conf.clmethod+'/'+conf.res_folder

    print(conf.device)
    if opt.device.startswith("cuda"):
        print("have ", torch.cuda.device_count(), "GPU.")
    gc.collect()
    torch.cuda.empty_cache()

    ##=== set random seed
    set_seed(opt, conf.seed)

    ##=== load tokenizer and tokenize the data
    if conf.encoder!='bilstm':
        tokenizer = context_models[conf.encoder]["tokenizer"].from_pretrained(conf.encoder, add_prefix_space=True)
    else:
        tokenizer = None
        build_word_idx(conf)
        read_pretrain_embedding(conf)
        build_emb_table(conf) # get word embedding

    _read_txt('../data/'+conf.dataset+'/train.txt', '../data/'+conf.dataset+'/'+conf.encoder+'_train.json', tokenizer, conf.digit2zero, conf)
    _read_txt('../data/'+conf.dataset+'/dev.txt', '../data/'+conf.dataset+'/'+conf.encoder+'_dev.json', tokenizer, conf.digit2zero, conf)
    _read_txt('../data/'+conf.dataset+'/test.txt', '../data/'+conf.dataset+'/'+conf.encoder+'_test.json', tokenizer, conf.digit2zero, conf)

    logger.info("Building a mapping between label string and its integer encoding.")

    ##=== load all data and tokenize the labels
    get_labelids(conf) # config.label2idx; config.idx2label
    
    ##=== load data
    # load data iterator
    trains = (
        MapStyleJsonDataset('../data/'+conf.dataset+'/'+conf.encoder+'_train.json', conf.label2idx)
        if conf.train_file else None
    )
    devs = (
        MapStyleJsonDataset('../data/'+conf.dataset+'/'+conf.encoder+'_dev.json', conf.label2idx)
        if conf.dev_file else None
    )
    tests = (
        MapStyleJsonDataset('../data/'+conf.dataset+'/'+conf.encoder+'_test.json', conf.label2idx)
        if conf.test_file else None
    )
    os.remove('../data/'+conf.dataset+'/'+conf.encoder+'_train.json')
    os.remove('../data/'+conf.dataset+'/'+conf.encoder+'_dev.json')
    os.remove('../data/'+conf.dataset+'/'+conf.encoder+'_test.json')
    data_collator = Collator(tokenizer, conf.device, conf.label_size)
    
    ##=== subset/shuffle the data
    train_ids = list(range(len(trains)))
    if conf.dataset!='wikigold_self':
        random.shuffle(train_ids)
    if conf.train_num>1:
        trains.setids(train_ids[:conf.train_num])
    else:
        trains.setids(train_ids)

    if conf.test_num>1:
        tests.setids(list(range(conf.test_num)))

    if conf.dev_num>1:
        devs.setids(list(range(conf.dev_num)))
    
    ##=== reveal some clean labels for fair comparison with useclean
    if conf.injectclean:
        cleanprop = conf.cleanprop
        num_clean = max(10, int(len(trains)*cleanprop))
        index_clean = random.sample(range(len(trains)),num_clean)        
        for i in index_clean:
            trains.update(i,'labels_id', trains[i]['gold_labels_id'])
            trains.update(i, 'labels', trains[i]['gold_labels'])
    
    ##=== compute some meta statistics related to the data
    # golden noise rate
    neg_noise_rate_gold, pos_noise_rate_gold = ratio_estimation_iterator(conf, trains)
    conf.neg_noise_rate_gold = neg_noise_rate_gold
    conf.pos_noise_rate_gold = pos_noise_rate_gold
    print('negative noise rate: '+str(neg_noise_rate_gold))
    print('positve noise rate: '+str(pos_noise_rate_gold))
    # observed entity class frequency
    entlist = [trains[i]['labels_id'] for i in range(len(trains.ids))]
    entlist = flatten(entlist)
    freqentlist = dict(collections.Counter(entlist))
    conf.ent_freq = [freqentlist[e] if e in freqentlist else 1 for e in range(conf.label_size)]
    weights = (1.0 - 0.9) / np.array(1.0 - np.power(0.9, conf.ent_freq))
    conf.ent_freq = weights / np.sum(weights) * conf.label_size
    
    ##=== train and save the results if not existed
    if os.path.exists(conf.res_folder+'/metrics.pkl'):
       print("The results exists. Loading exists results....")
       f = open(conf.res_folder+'/metrics.pkl', 'rb')
       allmetrics = pickle.load(f)
       f.close()
       outer_metrics = allmetrics[0] 
       inner_metrics = allmetrics[1]
    else:
        if conf.clmethod == 'none':
        # baseline, without any confident learning    
            inner_metrics = train_baseline(config=conf, data_collator = data_collator, train_insts = trains, dev_insts = devs, test_insts = tests)
            outer_metrics = None 

        if conf.clmethod == 'coreg': 
        # add regularization based on model aggreement on top of baseline     
            inner_metrics = train_coreg(config=conf, data_collator = data_collator, train_insts = trains, dev_insts = devs, test_insts = tests)
            outer_metrics = None
        
        if conf.clmethod == 'metaweight':  
        # weight on the utterance level through learned weights     
            inner_metrics = train_mw(config=conf, data_collator = data_collator, train_insts = trains, dev_insts = devs, test_insts = tests)
            outer_metrics = None 

        if conf.clmethod == 'CLin': 
        # marginalize noisy set of instances in the loss, 
        # whether noisy or not are determined using cutoff on confidence score
        # our useclean method is built in this as well
            inner_metrics = train_CLin(config=conf, data_collator = data_collator, train_insts = trains, dev_insts = devs, test_insts = tests)
            outer_metrics = None
            
        if conf.clmethod == 'CLout': 
        # use CLin to learn two models on two folds, then use one model to modify the other fold
            inner_metrics = train_CLout(config=conf, data_collator= data_collator, train_insts=trains, dev_insts=devs, test_insts=tests)
            outer_metrics = None              
              
        if conf.num_epochs > 0:
        # save all the metrics
            allmetrics = [outer_metrics, inner_metrics]
            f = open(conf.res_folder+'/metrics.pkl', 'wb')
            pickle.dump(allmetrics, f)
            f.close()    
        
    if conf.num_epochs > 0:

        if conf.clmethod == 'CLout':
            neg_noise_rate_gold, pos_noise_rate_gold = ratio_estimation_iterator(conf, trains)
            inner_metrics['neg_noise_rate_gold'] = neg_noise_rate_gold
            inner_metrics['pos_noise_rate_gold'] = pos_noise_rate_gold
            # also compute the noise rate after correction
            
            f = open(conf.res_folder+'/train_modified.pkl', 'rb')
            modified = pickle.load(f)
            f.close()
            modified = list(modified)
            neg_noise_rate_after, pos_noise_rate_after = ratio_estimation_iterator(conf, modified)
            inner_metrics['neg_noise_rate_after'] = neg_noise_rate_after
            inner_metrics['pos_noise_rate_after'] = pos_noise_rate_after
        
        print(inner_metrics)

        ##=== evaluate the best model (selected using dev_f1) saved on the test data
        with open(conf.res_folder+'/allresults.txt', 'w') as f:
            bestdev = -1
            if conf.cutoff in ['fitmix','fake','goracle']:
                f.write("%s %.5f \n" % ("poscut", conf.poscutoff))
                f.write("%s %.5f \n" % ("negcut", conf.negcutoff))  
                print('poscut: '+str(conf.poscutoff))
                print('negcut: '+str(conf.negcutoff))
            
            if isinstance(inner_metrics['dev_f1'][0], list):
                for itr, inner_metric in enumerate(inner_metrics['dev_f1']):
                    bestdev_cur = np.nanmax(inner_metric)
                    if bestdev_cur >= bestdev:
                        bestdev = bestdev_cur
                        epoch_id = np.nanargmax(inner_metric)
                        itr_id = itr
            else:
                bestdev_cur = np.nanmax(inner_metrics['dev_f1'])
                print(bestdev_cur)
                if bestdev_cur >= bestdev:
                    bestdev = bestdev_cur
                    epoch_id = np.nanargmax(inner_metrics['dev_f1'])
            
            for key, val in inner_metrics.items():
                if isinstance(val, list):
                    if isinstance(val[0], list):
                        value = val[itr_id][epoch_id]
                    else:
                        value = val[epoch_id]    
                if isinstance(val, float):
                    value = val        
                f.write("%s %.5f \n" % (key, value))
                  
        f.close()        

        ##==== plot the training process
        # plot inner metrics
        plotinnermetrics(metricslist = ['precision','recall','f1'], inner_metrics = inner_metrics, info='total', conf = conf)
        plotinnermetrics(metricslist = ['hard_precision','hard_recall','hard_f1'], inner_metrics = inner_metrics,  info='hard', conf = conf)
        plotinnermetrics(metricslist = ['easy_precision','easy_recall','easy_f1'], inner_metrics = inner_metrics,  info='easy', conf = conf)
        plotinnermetrics(metricslist = ['head_precision','head_recall','head_f1'], inner_metrics = inner_metrics,  info='head', conf = conf)
        plotinnermetrics(metricslist = ['tail_precision','tail_recall','tail_f1'], inner_metrics = inner_metrics,  info='tail', conf = conf)
        plotinnermetrics(metricslist = ['high_precision','high_recall','high_f1'], inner_metrics = inner_metrics, info='highfreq', conf = conf)
        plotinnermetrics(metricslist = ['low_precision','low_recall','low_f1'], inner_metrics = inner_metrics,  info='lowfreq', conf = conf)
        plotinnermetrics(metricslist = ['train_loss'], inner_metrics = inner_metrics, info='loss', conf = conf)
        if conf.diagonosis:
            plotinnermetrics(metricslist = ['train_noise_precision','train_noise_recall','train_noise_f1'], inner_metrics = inner_metrics, info='noise', conf = conf)
            plotinnermetrics(metricslist = ['train_clean_precision','train_clean_recall','train_clean_f1'], inner_metrics = inner_metrics, info='clean', conf = conf)

        if conf.clmethod.startswith('CL'):
            plotinnermetrics(metricslist = ['train_conf_precision','train_conf_recall','train_conf_f1'], inner_metrics = inner_metrics, info='totalconf', conf = conf)
            plotinnermetrics(metricslist = ['train_conf_neg_precision','train_conf_neg_recall','train_conf_neg_f1'], inner_metrics = inner_metrics,  info='negconf', conf = conf)
            plotinnermetrics(metricslist = ['train_conf_pos_precision','train_conf_pos_recall','train_conf_pos_f1'], inner_metrics = inner_metrics,  info='posconf', conf = conf)
            if conf.cutoff == 'oracle':
                plotinnermetrics(metricslist = ['train_conf_fit_precision','train_conf_fit_recall','train_conf_fit_f1'], inner_metrics = inner_metrics, info='totalconf', conf = conf)
                plotinnermetrics(metricslist = ['train_conf_fit_neg_precision','train_conf_fit_neg_recall','train_conf_fit_neg_f1'], inner_metrics = inner_metrics,  info='negconf', conf = conf)
                plotinnermetrics(metricslist = ['train_conf_fit_pos_precision','train_conf_fit_pos_recall','train_conf_fit_pos_f1'], inner_metrics = inner_metrics,  info='posconf', conf = conf)
            
            if conf.clmethod in ['CL','CLout']:
                plotinnermetrics(metricslist = ['before_precision','before_recall','before_f1'], inner_metrics = inner_metrics, info='before_total', conf = conf)
                plotinnermetrics(metricslist = ['before_hard_precision','before_hard_recall','before_hard_f1'], inner_metrics = inner_metrics,  info='before_hard', conf = conf)
                plotinnermetrics(metricslist = ['before_easy_precision','before_easy_recall','before_easy_f1'], inner_metrics = inner_metrics,  info='before_easy', conf = conf)
                plotinnermetrics(metricslist = ['before_head_precision','before_head_recall','before_head_f1'], inner_metrics = inner_metrics, info='before_head', conf = conf)
                plotinnermetrics(metricslist = ['before_tail_precision','before_tail_recall','before_tail_f1'], inner_metrics = inner_metrics,  info='before_tail', conf = conf)
                plotinnermetrics(metricslist = ['before_high_precision','before_high_recall','before_high_f1'], inner_metrics = inner_metrics, info='before_highfreq', conf = conf)
                plotinnermetrics(metricslist = ['before_low_precision','before_low_recall','before_low_f1'], inner_metrics = inner_metrics,  info='before_lowfreq', conf = conf)
                plotinnermetrics(metricslist = ['before_train_loss'], inner_metrics = inner_metrics, info='before_loss', conf = conf)


    
#==== helper function to plot in group
def plotinnermetrics(metricslist, inner_metrics, info = '', conf = Config):
    
    if 'test_'+metricslist[0] in inner_metrics:
        alllen = len(flatten(inner_metrics['test_'+metricslist[0]]))
    elif metricslist[0] in inner_metrics:
        alllen = len(flatten(inner_metrics[metricslist[0]]))
    else:
        return

    fig = plt.figure(figsize=(2*int(conf.num_outer_iterations*len(metricslist)/2),3))
    cnt = 0
    for metric in metricslist:
        cnt += 1
        ax = fig.add_subplot(1, int(len(metricslist)), cnt)
        if 'test_'+metric in inner_metrics:
            ax.plot(np.arange(alllen)+1, flatten(inner_metrics['test_'+metric]), alpha=.6, label = 'test')
            ax.plot(np.arange(alllen)+1, flatten(inner_metrics['dev_'+metric]), alpha=.6, label = 'dev')
            #maxy = np.nanmax(flatten(inner_metrics['test_'+metric])+flatten(inner_metrics['dev_'+metric]))
            if 'train_'+metric in inner_metrics:
                ax.plot(np.arange(alllen)+1, flatten(inner_metrics['train_'+metric]), alpha=.6, label = 'train')
                #maxy = np.nanmax(maxy, np.nanmax(flatten(inner_metrics['train_'+metric])))
        else:
            ax.plot(np.arange(alllen)+1, flatten(inner_metrics[metric]), alpha=.6)
            #maxy = np.nanmax(flatten(inner_metrics[metric]))

        ax.set_xlabel('Total Epochs')
        ax.set_ylabel(metric)
        if conf.clmethod!='none':
            ax.set_title('{}'.format(metric))
        else:
            ax.set_title('{}'.format(metric))
        if alllen == (conf.num_epochs+1)*conf.num_outer_iterations:
            ax.vlines(x=(conf.num_epochs+1)*np.arange(conf.num_outer_iterations+1), 
                   ymin=0, ymax=100,colors='black', lw=1, linestyle='dotted')
        #ax1.hlines(x=baseprecision, 
        #           xmin=0, xmax=len(devprecision)+1,colors='red', lw=1, linestyle='dotted')
        ax.legend()   
    fig.tight_layout()
    plt.savefig(conf.res_folder+"/"+info+"_inneriter.png", format="png")
    plt.close()


def plotoutermetrics(metricslist, outer_metrics, info = '', conf = Config):
    fig = plt.figure(figsize=(2*int(len(metricslist)),3))
    cnt = 0
    for metric in metricslist:
        cnt += 1
        ax = fig.add_subplot(1, int(len(metricslist)), cnt)
        if 'test_'+metric in outer_metrics:
            ax.plot(np.arange(conf.num_outer_iterations), flatten(outer_metrics['test_'+metric]), alpha=.6, label = 'test')
            ax.plot(np.arange(conf.num_outer_iterations), flatten(outer_metrics['dev_'+metric]), alpha=.6, label = 'dev')
            if 'train_'+metric in outer_metrics:
                ax.plot(np.arange(conf.num_outer_iterations), flatten(outer_metrics['train_'+metric]), alpha=.6, label = 'train')
        else:
            ax.plot(np.arange(conf.num_outer_iterations), flatten(outer_metrics[metric]), alpha=.6)
        ax.set_xlabel('Outer Iterations')
        ax.set_ylabel(metric)
        if conf.clmethod!='none':
            ax.set_title('{}'.format(metric))
        else:
            ax.set_title('{}'.format(metric))
                 
        ax.legend()
    fig.tight_layout()
    plt.savefig(conf.res_folder+"/"+info+"_outiter.png", format="png")    


if __name__ == "__main__":
    main()
