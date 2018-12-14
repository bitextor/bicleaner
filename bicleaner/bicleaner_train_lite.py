#!/usr/bin/env python

from heapq import heappush, heappop
from mosestokenizer import MosesTokenizer
from sklearn import neighbors
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from tempfile import TemporaryFile, NamedTemporaryFile
from timeit import default_timer

import argparse
import logging
import math
import numpy as np
import os
import random
import sklearn
import sys

#Allows to load modules while inside or outside the package  
try:
    from .features import feature_extract
    from .prob_dict import ProbabilisticDictionary
    from .util import no_escaping, check_positive, check_positive_or_zero, logging_setup
except (SystemError, ImportError):
    from features import feature_extract
    from prob_dict import ProbabilisticDictionary
    from util import no_escaping, check_positive, check_positive_or_zero, logging_setup    

__author__ = "Sergio Ortiz-Rojas"
# Please, don't delete the previous descriptions. Just add new version description at the end.
__version__ = "Version 0.1 # December 2017 # Initial version # Sergio Ortiz-Rojas"
__version__ = "Version 0.2 # 09/01/2018 # Adding argument for injecting wrong examples from a file # Jorge Ferrández-Tordera"
__version__ = "Version 0.3 # 14/12/2018 # Lite version # Marta Bañón"

# Calculate precision, recall and accuracy over the 0.0,1.0,0.1 histogram of
# good and  wrong alignments
def precision_recall(hgood, hwrong):
    precision = []
    recall    = []
    accuracy  = []
    total = sum(hgood) + sum(hwrong)

    for i in range(len(hgood)):
        tp = sum(hgood[i:])   # true positives
        fp = sum(hwrong[i:])  # false positives
        fn = sum(hgood[:i])   # false negatives
        tn = sum(hwrong[:i])  # true negatives
        try:
            precision.append(tp*1.0/(tp+fp))     # precision = tp/(tp+fp)
        except ZeroDivisionError:
            precision.append(math.nan)
        try:
            recall.append(tp*1.0/(tp+fn))        # recall = tp/(tp+fn)
        except ZeroDivisionError:
            recall.append(math.nan)
        try:
            accuracy.append((tp+tn)*1.0/total)   # accuracy = (tp+tn) / total
        except ZeroDivisionError:
            accuracy.append(math.nan)

    return precision, recall, accuracy


def repr_right(numeric_list, numeric_fmt = "{:1.7f}"):
    result_str = ["["]
    for i in range(len(numeric_list)):
        result_str.append(numeric_fmt.format(numeric_list[i]))
        if i < (len(numeric_list)-1):
            result_str.append(", ")
        else:
            result_str.append("]")
    return "".join(result_str)
    
# Write YAML with the training parameters and quality estimates
def write_metadata(myargs, length_ratio, hgood, hwrong):
    out = myargs.metadata

    precision, recall, accuracy = precision_recall(hgood, hwrong)

    # Writing it by hand (not using YAML libraries) to preserve the order
    out.write("classifier: {}\n".format(os.path.abspath(myargs.classifier.name)))
    out.write("classifier_type: {}\n".format(myargs.classifier_type))
    out.write("source_lang: {}\n".format(myargs.source_lang))
    out.write("target_lang: {}\n".format(myargs.target_lang))
    out.write("source_dictionary: {}\n".format(os.path.abspath(myargs.source_dictionary.name)))
    out.write("target_dictionary: {}\n".format(os.path.abspath(myargs.target_dictionary.name)))
    out.write("normalize_by_length: {}\n".format(myargs.normalize_by_length))
    out.write("treat_oovs: {}\n".format(myargs.treat_oovs))
    out.write("qmax_limit: {}\n".format(myargs.qmax_limit))
    out.write("disable_features_quest: {}\n".format(myargs.disable_features_quest))
    out.write("good_examples: {}\n".format(myargs.good_examples))
    out.write("wrong_examples: {}\n".format(myargs.wrong_examples))
    out.write("good_test_examples: {}\n".format(myargs.good_test_examples))
    out.write("wrong_test_examples: {}\n".format(myargs.wrong_test_examples))
    out.write("good_test_histogram: {}\n".format(hgood.__repr__()))
    out.write("wrong_test_histogram: {}\n".format(hwrong.__repr__()))
    out.write("precision_histogram: {}\n".format(repr_right(precision)))
    out.write("recall_histogram: {}\n".format(repr_right(recall)))
    out.write("accuracy_histogram: {}\n".format(repr_right(accuracy)))
    out.write("length_ratio: {:1.7f}\n".format(length_ratio))
    
# Argument parsing
def initialization():
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
    groupO.add_argument('--normalize_by_length', action='store_true', help="Normalize by length in qmax dict feature")
    groupO.add_argument('--treat_oovs', action='store_true', help="Special treatment for OOVs in qmax dict feature")
    groupO.add_argument('--qmax_limit', type=check_positive_or_zero, default=20, help="Number of max target words to be taken into account, sorted by length")
    groupO.add_argument('--disable_features_quest', action='store_false', help="Disable less important features")
    groupO.add_argument('-g', '--good_examples',  type=check_positive_or_zero, default=50000, help="Number of good examples")
    groupO.add_argument('-w', '--wrong_examples', type=check_positive_or_zero, default=50000, help="Number of wrong examples")
    groupO.add_argument('--good_test_examples',  type=check_positive_or_zero, default=10000, help="Number of good test examples")
    groupO.add_argument('--wrong_test_examples', type=check_positive_or_zero, default=10000, help="Number of wrong test examples")
    groupO.add_argument('--classifier_type', choices=['svm', 'nn', 'nn1', 'adaboost', 'random_forest'], default="random_forest", help="Classifier type")
    groupO.add_argument('--dump_features', type=argparse.FileType('w'), default=None, help="Dump training features to file")
    groupO.add_argument('--wrong_examples_file', type=argparse.FileType('r'), default=None, help="File with wrong examples extracted to replace the synthetic examples from method used by default")

    groupL = parser.add_argument_group('Logging')
    groupL.add_argument('-q', '--quiet', action='store_true', help='Silent logging mode')
    groupL.add_argument('--debug', action='store_true', help='Debug logging mode')
    groupL.add_argument('--logfile', type=argparse.FileType('a'), default=sys.stderr, help="Store log to a file")

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

    return hgood[0].tolist(), hwrong[0].tolist()

# Random shuffle corpora to ensure fairness of training and estimates.
def shuffle(input, n_aligned, n_misaligned, wrong_examples_file):
    logging.info("Shuffle starts")
    good_sentences  = TemporaryFile("w+")
    wrong_sentences = TemporaryFile("w+")
    total_size   = 0
    length_ratio = 0

    with TemporaryFile("w+") as temp:
        logging.info("Indexing file")
        # (1) Calculate the number of lines, length_ratio, offsets
        offsets = []
        nline = 0
        ssource = 0
        starget = 0
        count = 0

        for line in input:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                offsets.append(count)
                count += len(bytearray(line, "UTF-8"))
                ssource += len(parts[0])
                starget += len(parts[1])
                nline += 1
                temp.write(line)

        temp.flush()
        
        total_size = nline

        if total_size == 0:
            raise Exception("The input file {} is empty".format(input.name))
        elif not wrong_examples_file and  total_size < max(n_aligned, n_misaligned):
            raise Exception("Aborting... The input file {} has less lines than required by the numbers of good ({}) and wrong ({}) examples. Total lines required: {}".format(input.name, n_aligned, n_misaligned, n_aligned + n_misaligned))

        try:
            length_ratio = (ssource * 1.0)/(starget * 1.0) # It was (starget * 1.0)/(ssource * 1.0)
        except ZeroDivisionError:
            length_ratio = math.nan

        logging.info("Shuffling good sentences")
        # (2) Get good sentences
        random.shuffle(offsets)

        for i in offsets[0:n_aligned]:
            temp.seek(i)
            good_sentences.write(temp.readline())

        logging.info("Shuffling wrong sentences")
        # (3) Get wrong sentences
        if wrong_examples_file:
            # The file is already shuffled
            logging.info("Using wrong examples from file {} instead the synthetic method".format(wrong_examples_file.name))
            
            count = 0
            for i in wrong_examples_file:
                wrong_sentences.write(i)
                count += 1
                if count == n_misaligned:
                    break
            
            
            
        else:
            wrong_lines = min(total_size, n_misaligned)
            if (wrong_lines > 0):
                offsets_copy = offsets[:]
                random.shuffle(offsets)
                random.shuffle(offsets_copy)
                for i in range(wrong_lines):
                    temp.seek(offsets[i])
                    line = temp.readline()
                    parts = line.rstrip("\n").split("\t")
                    wrong_sentences.write(parts[0])
                    wrong_sentences.write("\t")
                    temp.seek(offsets_copy[i])
                    line = temp.readline()
                    parts = line.rstrip("\n").split("\t")
                    wrong_sentences.write(parts[1])
                    wrong_sentences.write("\n")
            else:
                logging.warning("Number of misaligned examples is 0")
        temp.close()
    logging.info("Shuffling ends")

    good_sentences.seek(0)
    wrong_sentences.seek(0)

    return total_size, length_ratio, good_sentences, wrong_sentences




# Main loop of the program
def perform_training(args):
    global nline
    time_start = default_timer()
    logging.info("Starting process")


    # Shuffle and get length ratio
    total_size, length_ratio, good_sentences, wrong_sentences = shuffle(args.input, args.good_examples + args.good_test_examples, args.wrong_examples + args.wrong_test_examples, args.wrong_examples_file)
    args.length_ratio = length_ratio
     

    # Load dictionaries
    args.dict_sl_tl = ProbabilisticDictionary(args.source_dictionary)
    args.dict_tl_sl = ProbabilisticDictionary(args.target_dictionary)
    
    features_file = NamedTemporaryFile( delete=False)
    
    with MosesTokenizer(args.source_lang) as tokl, \
         MosesTokenizer(args.target_lang) as tokr:
        with open(good_sentences.name, 'r') as gsf, \
        open(wrong_sentences.name, 'r') as wsf, \
        open(features_file.name, 'w+') as fileout:

            for i in gsf:
                srcsen,trgsen = i.split("\t")[:2]
#                print(str(i) + " ---" + str(srcsen) + " --- " + str(trgsen))
                features = feature_extract(srcsen, trgsen, tokl, tokr, args)
                for j in features:
                    fileout.write("{}".format(j))
                    fileout.write("\t")
<<<<<<< HEAD
                fileout.write("{}".format(1))
=======
                fileout.write("{}".format(label))
>>>>>>> e75b2f233990749efc64f5afb67c2f6f27ed4105
                fileout.write("\n")
            fileout.flush()    
            
            for i in wsf:
                srcsen,trgsen = i.split("\t")[:2]
#                print(str(i) + " ---" + str(srcsen) + " --- " + str(trgsen))
                features = feature_extract(srcsen, trgsen, tokl, tokr, args)
                for j in features:
                    fileout.write("{}".format(j))
                    fileout.write("\t")
                fileout.write("{}".format(0))
                fileout.write("\n")
            fileout.flush()        

       
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
        hgood, hwrong = train_classifier(features_train, features_test, args.classifier_type, args.classifier)
        features_train.close()
        features_test.close()

    logging.info("End training")

    write_metadata(args, length_ratio, hgood, hwrong)
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
