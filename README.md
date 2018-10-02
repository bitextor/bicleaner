# bicleaner

Bicleaner is a tool in Python that allows to classify a parallel corpus, 
indicating the likelihood of a pair of sentences being mutual translations (with a value near to 1) or not  (with a value near to 0)

Although a training script (bicleaner-train.py) is provided, you may want to use the available ready-to-use language packages. 
Please, visit https://github.com/bitextor/bitextor-data/tree/master/bicleaner to download the language packages and the documentation on how to use them 
(TL;DR: download the desired language package (i.e. `en-de.tar.gz`) and uncompress it (`tar -xzvf en-de.tar.gz`) in the folder "`lang`" in bicleaner's sourcecode folder)

## Citation 

If you find Bicleaner useful, please consider citing the following paper:

> V. M. Sánchez-Cartagena, M. Bañón, S. Ortiz-Rojas and G. Ramírez-Sánchez,\
> "Prompsit's submission to WMT 2018 Parallel Corpus Filtering shared task",\
>in *Proceedings of the Third Conference on Machine Translation, Volume 2: Shared Task Papers*.\
>Brussels, Belgium: Association for Computational Linguistics, October 2018

```
@InProceedings{prompsit:2018:WMT,
  author    = { V\'{i}ctor M. S\'{a}nchez-Cartagena and Marta Ba{\~n}\'{o}n and Sergio Ortiz-Rojas and Gema Ram\'{i}rez-S\'{a}nchez},
  title     = {Prompsit's submission to WMT 2018 Parallel Corpus Filtering shared task},
  booktitle = {Proceedings of the Third Conference on Machine Translation, Volume 2: Shared Task Papers},
  month     = {October},
  year      = {2018},
  address   = {Brussels, Belgium},
  publisher = {Association for Computational Linguistics}
}
```

## Requirements

Bicleaner works with Python3 only. Dependences for Bicleaner are in the requirements.txt file and can be installed with `pip`:

`python3 -m pip install -r requirements.txt`


## Cleaning

bicleaner-classifier-full.py is a Python script that allows to classify a parallel corpus, 
indicating the likelihood of a pair of sentences being mutual translations (with a value near to 1) or not  (with a value near to 0)

The input file must contain at least four columns:

* col1: URL 1
* col2: URL 2
* col3: Source sentence
* col4: Target sentence

The generated output file will contain the same lines and columns that the original input file, 
adding an extra column containing the Bicleaner classifier score.

This script can be run with

```
python3 bicleaner-classifier-full.py [-h] -m METADATA [-s SOURCE_LANG]
                                    [-t TARGET_LANG] [--tmp_dir TMP_DIR]
                                    [-b BLOCK_SIZE] [-p PROCESSES]
                                    [--normalize_by_length] [--treat_oovs]
                                    [--qmax_limit QMAX_LIMIT]
                                    [--disable_features_quest]
                                    [-g GOOD_EXAMPLES] [-w WRONG_EXAMPLES]
                                    [--good_test_examples GOOD_TEST_EXAMPLES]
                                    [--wrong_test_examples WRONG_TEST_EXAMPLES]
                                    [-d DISCARDED_TUS] [--threshold THRESHOLD]
                                    [-q] [--debug] [--logfile LOGFILE] [-v]
                                    input [output]
```

### Parameters

* positional arguments:
  * input: Tab-separated files to be classified  (line format: URL1 URL2 SOURCE_SENTENCE TARGET_SENTENCE, tab-separated)
  * output: Output of the classification (default: standard output)
* optional arguments:
  * -h, --help: show this help message and exit
* Mandatory:
  * -m METADATA, --metadata METADATA: Training metadata (YAML file). Take into account that explicit command line arguments will overwrite the values from metadata file (default: None)
* Optional:
  * --tmp_dir TMP_DIR: Temporary directory where creating the temporary files of this program (default: user's temp dir)
  * -b BLOCK_SIZE, --block_size BLOCK_SIZE Sentence pairs per block (default: 10000)
  * -p PROCESSES, --processes PROCESSES: Number of processes to use (default: all CPUs minus one)
  * --normalize_by_length: Normalize by length in qmax dict feature 
  * --treat_oovs: Special treatment for OOVs in qmax dict feature
  * --qmax_limit: Number of max target words to be taken into account, sorted by length (default: 20)
  * --disable_features_quest: Disable less important features
  * -g GOOD_EXAMPLES, --good_examples GOOD_EXAMPLES: Number of good examples (default: 50000)
  * -w WRONG_EXAMPLES, --wrong_examples WRONG_EXAMPLES: Number of wrong examples (default: 50000)
  * --good_test_examples GOOD_TEST_EXAMPLES: Number of good test examples (default: 2000)
  * --wrong_test_examples WRONG_TEST_EXAMPLES: Number of wrong test examples (default: 2000)
  * -d DISCARDED_TUS, --discarded_tus DISCARDED_TUS: TSV file with discarded TUs. Discarded TUs by the classifier are written in this file in TSV file. (default: None)
  * --threshold THRESHOLD: Threshold for classifier. If accuracy histogram is present in metadata, the interval for max value will be given as a default instead the current default. (default: 0.5)
* Logging:
  * -q, --quiet: Silent logging mode (default: False)
  * --debug: Debug logging mode (default: False)
  * --logfile LOGFILE: Store log to a file (default: <_io.TextIOWrapper name='<stderr>' mode='w' encoding='UTF-8'>)
  * -v, --version: show version of this script and exit

### Example

```
python3 bicleaner-classifier-full.py  \
        corpus.en-es.tabs  \
        classifiedCorpora/en-es/corpus.en-es.tabs.classifed  \
        -m lang/en-es/training.en-es.yaml \
        -b 100  \
        --tmp_dir /home/user/tmp 
```

This will read the "corpus.en-es.tabs" file, 
classify it with the classifier indicated in the "training.en-es.yaml" metadata file,
writing the result of the classification in the "corpus.en-es.tabs.classified" file.
Each line of the new file will contain the same content as the input file, adding a column with the score given by the Bicleaner classifier.

### Test 

Running the provided testset is recommended to check that Bicleaner is working as expected. The testset can be run with Pytest:

```
python3 -m pytest  tests/bicleaner_test.py -s

```
If everything went ok, a "1 passed in xx.xx seconds" message will be shown.

## Training classifiers

In case you need to train a new language model (i.e. because it is not available in the language packs provided in bitextor-data), 
you can use the `bicleaner-train.py` tool .
`bicleaner-train.py` is a Python script that allows to train a classifier which predicts
whether a pair of sentences are mutual translations or not.
It can be used as follows:


```
 python3 bicleaner-train.py [-h] -m METADATA -c CLASSIFIER -s SOURCE_LANG -t
                          TARGET_LANG -d SOURCE_DICTIONARY -D
                          TARGET_DICTIONARY [--normalize_by_length]
                          [--treat_oovs] [--qmax_limit QMAX_LIMIT]
                          [--disable_features_quest] [-g GOOD_EXAMPLES]
                          [-w WRONG_EXAMPLES]
                          [--good_test_examples GOOD_TEST_EXAMPLES]
                          [--wrong_test_examples WRONG_TEST_EXAMPLES]
                          [--classifier_type {svm,nn,nn1,adaboost,random_forest}]
                          [--dump_features DUMP_FEATURES] [-b BLOCK_SIZE]
                          [-p PROCESSES]
                          [--wrong_examples_file WRONG_EXAMPLES_FILE] [-q]
                          [--debug] [--logfile LOGFILE]
                          [input]
```                          
              
### Parameters

* positional arguments:
  * input: Tab-separated bilingual input file (default: Standard input)
* optional arguments:
  * -h, --help: show this help message and exit
* Mandatory:
  * -m METADATA, --metadata METADATA: Training metadata (YAML file)
  * -c CLASSIFIER, --classifier CLASSIFIER: Classifier data file 
  * -s SOURCE_LANG, --source_lang SOURCE_LANG: Source language code 
  * -t TARGET_LANG, --target_lang TARGET_LANG: Target language code
  * -d SOURCE_DICTIONARY, --source_dictionary SOURCE_DICTIONARY: LR gzipped probabilistic dictionary 
  * -D TARGET_DICTIONARY, --target_dictionary TARGET_DICTIONARY: RL gzipped probabilistic dictionary
* Options:
  * --normalize_by_length: Normalize by length in qmax dict feature 
  * --treat_oovs: Special treatment for OOVs in qmax dict feature
  * --qmax_limit: Number of max target words to be taken into account, sorted by length (default: 20)
  * --disable_features_quest: Disable less important features 
  * -g GOOD_EXAMPLES, --good_examples GOOD_EXAMPLES: Number of good examples (default: 50000)
  * -w WRONG_EXAMPLES, --wrong_examples WRONG_EXAMPLES: Number of wrong examples (default: 50000)
  * --good_test_examples GOOD_TEST_EXAMPLES: Number of good test examples (default: 2000)
  * --wrong_test_examples WRONG_TEST_EXAMPLES: Number of wrong test examples (default: 2000)
  * --classifier_type {svm,nn,nn1,adaboost}: Classifier type (default: svm)
  * --dump_features DUMP_FEATURES: Dump training features to file (default: None)
  * -b BLOCK_SIZE, --block_size BLOCK_SIZE: Sentence pairs per block (default: 10000)
  * -p PROCESSES, --processes PROCESSES: Number of process to use (default: all CPUs minus one)
  * --wrong_examples_file WRONG_EXAMPLES_FILE: File with wrong examples extracted to replace the synthetic examples from method used by default (default: None)
* Logging:
  * -q, --quiet: Silent logging mode (default: False)
  * --debug: Debug logging mode (default: False)
  * --logfile LOGFILE: Store log to a file (default: <_io.TextIOWrapper name='<stderr>' mode='w' encoding='UTF-8'>)

### Example

```shell
$ python3 bicleaner-train.py \
          lang/en-cs/train.en-cs\
          --treat_oovs \
          --normalize_by_length \
          -s en \
          -t cs \
          -d lang/en-cs/en-cs.dict.gz \
          -D lang/en-cs/cs-en.dict.gz \
          -b  1000 \
          -c lang/en-cs/en-cs.classifier \
          -g 50000 \
          -w 50000 \
          -m lang/en-cs/training.en-cs.yaml \
          --classifier_type svm
```

This will train a SVM classifier for english-czech using the corpus train.en-cs and the probabilistic dictionaries en-cs.dict.gz and cs-en.dict.gz. 
This training will use 50000 good and 50000 bad examples, and a block size of 1000 sentences.
The classifier data will be stored in en-cs.classifier, with the metadata in training.en-cs.yaml.

The generated .yaml file provides the following information, that is useful to get a sense on how good or bad was the training:

```
classifier: lang/en-cs/en-cs.classifier
classifier_type: svm
source_lang: en
target_lang: cs
source_dictionary: lang/en-cs/en-cs.dict.gz
target_dictionary: lang/en-cs/cs-en.dict.gz
normalize_by_length: True
treat_oovs: True
qmax_limit: 20
disable_features_quest: True
good_examples: 50000
wrong_examples: 50000
good_test_examples: 2000
wrong_test_examples: 2000
good_test_histogram: [3, 7, 11, 18, 21, 32, 42, 68, 95, 703]
wrong_test_histogram: [1478, 105, 88, 63, 71, 48, 45, 47, 30, 25]
precision_histogram: [0.3333333333333333, 0.6563528637261357, 0.7036247334754797, 0.7484709480122325, 0.7832110839445803, 0.8281938325991189, 0.8606635071090047, 0.8946280991735537, 0.9355216881594373, 0.9656593406593407]
recall_histogram: [1.0, 0.997, 0.99, 0.979, 0.961, 0.94, 0.908, 0.866, 0.798, 0.703]
accuracy_histogram: [0.3333333333333333, 0.825, 0.8576666666666667, 0.8833333333333333, 0.8983333333333333, 0.915, 0.9203333333333333, 0.9213333333333333, 0.9143333333333333, 0.8926666666666667]
length_ratio: 0.9890133482780752
```
