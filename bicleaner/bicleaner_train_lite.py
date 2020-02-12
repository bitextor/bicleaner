#!/usr/bin/env python

from heapq import heappush, heappop
from sklearn import neighbors
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.externals import joblib
import joblib
from tempfile import TemporaryFile, NamedTemporaryFile
from timeit import default_timer
from toolwrapper import ToolWrapper
from mosestokenizer import MosesTokenizer

import argparse
import logging
import math
import numpy as np
import os
import random
import sklearn
import sys
import json

#Allows to load modules while inside or outside the package  
try:
    from .features import feature_extract, FEATURES_VERSION, Features
    from .prob_dict import ProbabilisticDictionary
    from .util import no_escaping, check_positive, check_positive_or_zero, logging_setup
    from .training import shuffle,precision_recall, repr_right, write_metadata, train_fluency_filter
except (SystemError, ImportError):
    from features import feature_extract, FEATURES_VERSION, Features
    from prob_dict import ProbabilisticDictionary
    from util import no_escaping, check_positive, check_positive_or_zero, logging_setup
    from training import shuffle,precision_recall, repr_right, write_metadata, train_fluency_filter

__author__ = "Sergio Ortiz-Rojas"
# Please, don't delete the previous descriptions. Just add new version description at the end.
__version__ = "Version 0.1 # December 2017 # Initial version # Sergio Ortiz-Rojas"
__version__ = "Version 0.2 # 09/01/2018 # Adding argument for injecting wrong examples from a file # Jorge Ferrández-Tordera"
__version__ = "Version 0.3 # 14/12/2018 # Lite version # Marta Bañón"
__version__ = "Version 0.4 # 18/12/2018 # Changed to generalized tokenizer # Marta Bañón"
__version__ = "Version 0.5 # 18/01/2019 # Integrated training of LM and refactor to avoid code duplicity # Víctor M. Sánchez-Cartagena"
__version__ = "Version 0.13 # 30/10/2019 # Features version 3  # Marta Bañón"


logging_level = 0
    
# Argument parsing
def initialization():

    global logging_level
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)

    parser.add_argument('input',  nargs='?', type=argparse.FileType('r'), default=sys.stdin,  help="Tab-separated bilingual input file")

    groupM = parser.add_argument_group("Mandatory")
    groupM.add_argument('-m', '--metadata', type=argparse.FileType('w'), required=True, help="Training metadata (YAML file)")
    groupM.add_argument('-c', '--classifier', type=argparse.FileType('wb'), required=True, help="Classifier data file")
    groupM.add_argument('-s', '--source_lang',  required=True, help="Source language code")
    groupM.add_argument('-t', '--target_lang', required=True, help="Target language code")
    groupM.add_argument('-d', '--source_dictionary',  type=argparse.FileType('r'), required=True, help="LR gzipped probabilistic dictionary")
    groupM.add_argument('-D', '--target_dictionary', type=argparse.FileType('r'), required=True, help="RL gzipped probabilistic dictionary")

    groupO = parser.add_argument_group('Options')
    groupO.add_argument('-S', '--source_tokeniser_path', help="Source language tokeniser absolute path")
    groupO.add_argument('-T', '--target_tokeniser_path', help="Target language tokeniser absolute path")
    groupO.add_argument('--normalize_by_length', action='store_true', help="Normalize by length in qmax dict feature")
    groupO.add_argument('--treat_oovs', action='store_true', help="Special treatment for OOVs in qmax dict feature")
    groupO.add_argument('--qmax_limit', type=check_positive_or_zero, default=20, help="Number of max target words to be taken into account, sorted by length")
    groupO.add_argument('--disable_features_quest', action='store_false', help="Disable less important features")
    groupO.add_argument('-g', '--good_examples',  type=check_positive_or_zero, default=50000, help="Number of good examples")
    groupO.add_argument('-w', '--wrong_examples', type=check_positive_or_zero, default=50000, help="Number of wrong examples")
    groupO.add_argument('--good_test_examples',  type=check_positive_or_zero, default=10000, help="Number of good test examples")
    groupO.add_argument('--wrong_test_examples', type=check_positive_or_zero, default=10000, help="Number of wrong test examples")
    groupO.add_argument('--classifier_type', choices=['svm', 'nn', 'nn1', 'adaboost', 'random_forest'], default="random_forest", help="Classifier type")
    groupO.add_argument('--dump_features', type=argparse.FileType('wb'), default=None, help="Dump training features to file")
    groupO.add_argument('--wrong_examples_file', type=argparse.FileType('r'), default=None, help="File with wrong examples extracted to replace the synthetic examples from method used by default")
    groupO.add_argument('--features_version', type=check_positive, default=FEATURES_VERSION , help="Version of the features")
    groupO.add_argument('--disable_lang_ident', default=False, action='store_true', help="Don't apply features that use language detecting")
    
    #For LM filtering
    groupO.add_argument('--noisy_examples_file_sl',  type=str, help="File with noisy text in the SL. These are used to estimate the perplexity of noisy text.")
    groupO.add_argument('--noisy_examples_file_tl',  type=str, help="File with noisy text in the TL. These are used to estimate the perplexity of noisy text.")
    groupO.add_argument('--lm_dev_size', type=check_positive_or_zero, default=2000, help="Number of sentences to be removed from clean text before training LMs. These are used to estimate the perplexity of clean text.")
    groupO.add_argument('--lm_file_sl', type=str, help="SL language model output file.")
    groupO.add_argument('--lm_file_tl', type=str, help="TL language model output file.")
    groupO.add_argument('--lm_training_file_sl', type=str, help="SL text from which the SL LM is trained. If this parameter is not specified, SL LM is trained from the SL side of the input file, after removing --lm_dev_size sentences.")
    groupO.add_argument('--lm_training_file_tl', type=str, help="TL text from which the TL LM is trained. If this parameter is not specified, TL LM is trained from the TL side of the input file, after removing --lm_dev_size sentences.")
    groupO.add_argument('--lm_clean_examples_file_sl', type=str, help="File with clean text in the SL. Used to estimate the perplexity of clean text. This option must be used together with --lm_training_file_sl and both files must not have common sentences. This option replaces --lm_dev_size.")
    groupO.add_argument('--lm_clean_examples_file_tl', type=str, help="File with clean text in the TL. Used to estimate the perplexity of clean text. This option must be used together with --lm_training_file_tl and both files must not have common sentences. This option replaces --lm_dev_size.")
    
    groupL = parser.add_argument_group('Logging')
    groupL.add_argument('-q', '--quiet', action='store_true', help='Silent logging mode')
    groupL.add_argument('--debug', action='store_true', help='Debug logging mode')
    groupL.add_argument('--logfile', type=argparse.FileType('a'), default=sys.stderr, help="Store log to a file")

    if logging_level <= logging.WARNING and logging_level != logging.DEBUG:
        logging.getLogger("MosesTokenizer").setLevel(logging.WARNING)
        logging.getLogger("MosesSentenceSplitter").setLevel(logging.WARNING)
        logging.getLogger("MosesPunctuationNormalizer").setLevel(logging.WARNING)

    args = parser.parse_args()
    # Logging
    logging_setup(args)
    return args



# Training function: receives two file descriptors, input and test, and a
# type classifiers and trains a classifier storing it in classifier_output
# and returns some quality estimates.
def train_classifier(input_features, test_features, classifier_type, classifier_output):
    feats=[]
    labels=[]

    # Load features and labels and format them as numpy array
    for line in input_features:
        parts=line.rstrip("\n").split("\t")

        feats.append( [float(v) for v in parts[:-1] ] )
        labels.append(int(parts[-1]))

    dataset = dict()
    dataset['data']   = np.array(feats)
    dataset['target'] = np.array(labels)
    
    # Train classifier
    if classifier_type == "svm":
        clf = svm.SVC(gamma=0.001, C=100., probability=True)
    elif classifier_type == "nn":
        clf = neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=1)
    elif classifier_type == "nn1":
        clf = neighbors.KNeighborsClassifier(n_neighbors=1, n_jobs=1)
    elif classifier_type == "adaboost":
        clf = AdaBoostClassifier(n_estimators=100)
    elif classifier_type == "random_forest":
        clf = RandomForestClassifier(bootstrap=True, class_weight=None,
                                     criterion='gini',
                                     max_depth=2, 
                                     max_features='auto', 
                                     max_leaf_nodes=None,
                                     min_impurity_decrease=0.0, 
                                     min_impurity_split=None,
                                     min_samples_leaf=1, 
                                     min_samples_split=2,
                                     min_weight_fraction_leaf=0.0, 
                                     n_estimators=200, n_jobs=1,
                                     oob_score=False, 
                                     random_state=0, 
                                     verbose=0, 
                                     warm_start=False)
    else:
        logging.error("Unknown classifier: "+ classifier_type)
        sys.exit(1)

    clf.fit(dataset['data'], dataset['target'])

    # Log sorted feature importances with their names
    if classifier_type in ('random_forest', 'adaboost'):
        feat_names = Features.cols + Features.optional
        feat_dict = dict(zip(feat_names, clf.feature_importances_))
        sorted_feat = {k: v for k, v in sorted(feat_dict.items(), key=lambda item: item[1])}
    else:
        sorted_feat = None

    joblib.dump(clf, classifier_output)

    feats = []
    labels = []

    for line in test_features:
        parts = line.rstrip("\n").split("\t")
        feats.append([float (v) for v in parts[:-1]])
        labels.append(int(parts[-1]))

    dataset = np.array(feats)
    prediction = clf.predict_proba(dataset)
    
    pos = 0
    good = []
    wrong = []
    for pred in prediction:
        if labels[pos] == 1:
           good.append(pred[1])
        else:
           wrong.append(pred[1])
        pos += 1

    hgood  = np.histogram(good,  bins = np.arange(0, 1.1, 0.1))
    hwrong = np.histogram(wrong, bins = np.arange(0, 1.1, 0.1))

    return hgood[0].tolist(), hwrong[0].tolist(), sorted_feat


# Main loop of the program
def perform_training(args):
    global nline
    time_start = default_timer()
    logging.info("Starting process")
    
    #Read input to a named temporary file
    #We may need to read it multiple times and that would be problematic if it is sys.stdin
    count_input_lines = 0
    input = NamedTemporaryFile(mode="w",delete=False)
    for line in args.input:
        input.write(line)
        count_input_lines = count_input_lines +1
    input.close()    

    if count_input_lines < 10000:
        logging.error("Training corpus must be at least 10K sentences long (was {}).".format(count_input_lines))
        sys.exit(1)
                    
    stats=None                
    with open(input.name) as input_f:
        args.input=input_f
        stats=train_fluency_filter(args)
        args.input.seek(0)

        # Shuffle and get length ratio
        total_size, length_ratio, good_sentences, wrong_sentences = shuffle(args.input, args.good_examples + args.good_test_examples, args.wrong_examples + args.wrong_test_examples, args.wrong_examples_file)
    os.remove(input.name)
    
    args.length_ratio = length_ratio
     

    # Load dictionaries
    args.dict_sl_tl = ProbabilisticDictionary(args.source_dictionary)
    args.dict_tl_sl = ProbabilisticDictionary(args.target_dictionary)
    
    features_file = NamedTemporaryFile( delete=False)
    if args.source_tokeniser_path:
        tokl = ToolWrapper(args.source_tokeniser_path.split(' '))
    else:
        tokl = MosesTokenizer(args.source_lang)
    if args.target_tokeniser_path:
        tokr = ToolWrapper(args.target_tokeniser_path.split(' '))
    else:
        tokr = MosesTokenizer(args.target_lang)
    with open(good_sentences.name, 'r') as gsf, \
    open(wrong_sentences.name, 'r') as wsf, \
    open(features_file.name, 'w+') as fileout:

        for i in gsf:
            srcsen,trgsen = i.split("\t")[:2]
#            print(str(i) + " ---" + str(srcsen) + " --- " + str(trgsen))
            features = feature_extract(srcsen, trgsen, tokl, tokr, args)
            for j in features:
                fileout.write("{}".format(j))
                fileout.write("\t")
            fileout.write("{}".format(1))
            fileout.write("\n")
        fileout.flush()    
        
        for i in wsf:
            srcsen,trgsen = i.split("\t")[:2]
#            print(str(i) + " ---" + str(srcsen) + " --- " + str(trgsen))
            features = feature_extract(srcsen, trgsen, tokl, tokr, args)
            for j in features:
                fileout.write("{}".format(j))
                fileout.write("\t")
            fileout.write("{}".format(0))
            fileout.write("\n")
        fileout.flush()        
    tokl.close()
    tokr.close()
   
    features_file.seek(0)
    
    
   
    if args.dump_features:
        logging.info("Dumping features to " + os.path.abspath(args.dump_features.name))
        for i in features_file:
            args.dump_features.write(i)
        args.dump_features.close()
        features_file.seek(0)

    logging.info("Start training")
    features_file.close()

    hgood = []
    hwrong = []
    with TemporaryFile("w+") as features_train, TemporaryFile("w+") as features_test, open(features_file.name, 'r') as ff:
        nline = 0
        for line in ff:
#            print(line)
            if nline < args.good_examples:
                features_train.write(line)
            elif nline < args.good_examples + args.good_test_examples:
                features_test.write(line)
            elif nline < args.good_examples + args.good_test_examples + args.wrong_examples:
                features_train.write(line)
            else:
                features_test.write(line)
            nline += 1
        
        features_train.flush()
        features_test.flush()
        
        features_train.seek(0)
        features_test.seek(0)
        hgood, hwrong, feat_importances = train_classifier(features_train, features_test, args.classifier_type, args.classifier)
        features_train.close()
        features_test.close()

    logging.info("End training")

    if feat_importances is not None:
        logging.debug('Feature importances: ' + json.dumps(feat_importances, indent=4))
    write_metadata(args, length_ratio, hgood, hwrong, stats)
    args.metadata.close()

    # Stats
    logging.info("Finished")
    elapsed_time = default_timer() - time_start
    logging.info("Elapsed time {:.2f} s".format(elapsed_time))

# Main function: setup logging and calling the main loop
def main(args):
    # Parameter parsing
#    args = initialization()

    # Filtering
    perform_training(args)

if __name__ == '__main__':
    args = initialization()
    main(args)
