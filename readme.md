# Noisyner-Confaug

Noisyner-Confaug is a python repo for NER denoising. It contains the following popular methods, including our novel methods. 
- [baseline](https://github.com/allanj/pytorch_neural_crf): Neural-CRF model without any de-noising.
- [NLNCE](https://github.com/liukun95/Noisy-NER-Confidence-Estimation): denoising NER with confidence estimation. 
- [CoReg](https://github.com/wzhouad/NLL-IE): add regularization term based on agreement among multiple models trained on parallel to increase robustness.
- [MetaWeightNet](https://github.com/LindgeW/MetaAug4NER): Use a one-layer network to learn how to weight different utterance, such that the differences of the learned distribution and clean meta data distribution is minized.
- [CutFake](https://github.com/Manuscrit/Area-Under-the-Margin-Ranking): assign some random tokens to an additional fake class and use the lower quantile of their confidence score for sample selection.
- [FilDist](https://github.com/yasumasaonoe/DenoiseET/blob/master/denoising_models.py): train a binary classifier on additional dataset with both clean and noisy labels for sample selection.
- [UseClean](https://www.overleaf.com/project/62d051dbdffe7b6f4d41d04a): Our method, which includes training a clean anchor model using minimal clean supervision and then do weighted semi-supervised learning with warm start.

We provide different backbone models to use:
- [BiLSTM](https://github.com/allanj/pytorch_neural_crf/blob/39c7fe67ec099908ca2e4f55b1bd585d0579322d/src/model/module/bilstm_encoder.py) with random or Glove embedding.
- [BERT](https://github.com/huggingface/transformers/tree/v4.3.0) with random or pretrained embedding.

We have also implemented some popular techniques to increase robustness and handling long-tail case:
- [Contrast](https://github.com/princeton-nlp/SimCSE): use contrastive learning to learn more robust hidden representation. 
- [Balance](https://github.com/google-research/google-research/tree/master/logit_adjustment): adding additional term based on entity class frequency to the logit, to empahasis more on under-represented entity class.

This repo also contains automatic evaluation script which includes recording the key statistics during whole training process and visualization of them.

All the code is under /src folder. 


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages.

```bash
pip3 install -r requirements.txt
```
## Data preparation
We provide all the datasets that we used in our project under the ./data repo. For running on new dataset, just convert the new dataset into format like the ones in the ./data repo: i.e. train.txt, dev.txt, and test.txt. We assume train.txt contains three columns, where the first one is words, the second one is noisy labels and the third one is gold labels. For dev.txt and test.txt, we assume they contain two columns, where the first one is words, and the second one gold labels.


## Usage
### Run single method for one dataset
Given dataset repo name "data", we can run the following method on via the corresponding command lines. 
- baseline: 
```bash
python3 main.py --dateset data --clmethod none
```
- NLNCE:
```bash
python3 main.py --dataset data --clmethod CLin --score nerloss --cutoff heuri
```
- MetaWeightNet:
```bash
python3 main.py --dataset data --clmethod metaweight --cleanprop 0.03
```
- CoReg:
```bash
python3 main.py --dataset data --clmethod coreg --alpha 3 --alpha_warmup_ratio 0.1
```
- CutFake:
```bash
python3 main.py --dataset data --clmethod CLin --score useclean --usecleanscore nerloss --cleanprop 0.03 --cutoff fake --warm true --weight true --numfake 10 --fakeq 0.3
```
- UseClean:
```bash
python3 main.py --dataset data --CLmethod CLin --score useclean --usecleanscore nerloss --cleanprop 0.03 --cutoff fitmix --warm true --weight true --usef1 true
```

### Run all methods for one dataset
```bash
source runone.sh cuda data seed info cleanprop numsamples 
```
Here `cuda` specifies the cuda device number, `data` specifies the dataset name, `seed` specify the random seed, `info` specifiy what info to add on these experiments, `cleanprop` specifiy how many proportion of clean data should we use, `numsamples` specify how many samples to use for training, dev and test (for quick run test). 
### Reproduce all experiments
To reproduce all experiments in our paper, simply using following command line with input of cuda device and random seed to use (we use seed 1 throughout our whole paper).
```bash
source runall.sh cuda seed
```
### Example data and results
We also include an toy example data `./data/massive_en_us__noise_bias_level1`, which is a noisy version of the Massive en-us dataset. We also include results of running the following core methods in `./results` folder, we organize the results of each single one into one independent folder, under which both the config, evaluation metrics are saved, as well as plots to visualize how different key metrics change during the trianing process. 
```
# baseline method with diagnosis process 
python3 main.py --dataest massive_en_us__noise_bias_level1 --clmethod none --diag true --seed 1 --info bert1
# NLNCE method with diagnosis process
python3 main.py --dateset massive_en_us__noise_bias_level1 --clmethod CLin --cutoff heuri --diag true --seed 1 --info bert1
# UseClean method with diagnosis process
python3 main.py --dateset massive_en_us__noise_bias_level1 --clmethod CLin --score useclean --usecleanprop 0.03 --warm true --weight true --cutoff fitmix --diag true --seed 1 --info bert1 # fitmix cutoff fitting
```


### Other functionalities
- change backbone encoder: we can change the backbone from BERT to BiLSTM using the ``--encoder bilstm`` argument; the default is ``bert-base-uncased``.

- change classificatio heads: we can change the classification heads from CRF (default) to MLP via ``--classifier mlp``. 

- change input embedding: for BERT, we can change from pretrained embedding (default) from random embedding via activating ``--random true``.

- add contrastive learning into training: we can add contrastive learning into the encoder training by active ``--contrastive true``.

- add logit adjustment into training: we can add logit adjustment into the encoder training by setting ``--tau 0.1`` or some number bigger than zero.

- to plot all the training dynamics: we can plot all the training dynamics as well as confidence score evaluation via setting ``--diag true``. 

- for our UseClean methods, to compute all the possible confidence scores using the clean anchor model (nerloss, encoderloss, diff, spikeness, entropy, aum), simply set ``--num_epochs 0`` will automatically activate this function. All the confidence scores will be saved in ``confscore_all.txt`` and plotted out for deep dive. 

- for our UseClean methods, to use adaptive sampling, simply change `--score useclean` to `--score usecleantail` or `--score usecleanhead` to sample more from tail or head entities. For definition of tail and head entities, see our papper.