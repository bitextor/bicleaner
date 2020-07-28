#!/usr/bin/env python

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

from heapq import heappush, heappop
from multiprocessing import Queue, Process, Value, cpu_count
from sklearn import neighbors
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import joblib
from sklearn import metrics
from tempfile import TemporaryFile, NamedTemporaryFile
from timeit import default_timer

import argparse
import logging
import numpy as np
import os
import random
import sys
import json

#Allows to load modules while inside or outside the package  
try:
    from .features import feature_extract, FEATURES_VERSION, Features
    from .prob_dict import ProbabilisticDictionary
    from .word_freqs_zipf import WordZipfFreqDist
    from .word_freqs_zipf_double_linked import WordZipfFreqDistDoubleLinked
    from .util import no_escaping, check_positive, check_positive_or_zero, logging_setup
    from .training import build_noisy_set, precision_recall, write_metadata, train_fluency_filter, train_porn_removal
    from .tokenizer import Tokenizer
except (SystemError, ImportError):
    from features import feature_extract, FEATURES_VERSION, Features
    from prob_dict import ProbabilisticDictionary
    from word_freqs_zipf import WordZipfFreqDist
    from word_freqs_zipf_double_linked import WordZipfFreqDistDoubleLinked
    from util import no_escaping, check_positive, check_positive_or_zero, logging_setup
    from training import build_noisy_set, precision_recall, write_metadata, train_fluency_filter, train_porn_removal
    from tokenizer import Tokenizer

__author__ = "Sergio Ortiz-Rojas"
# Please, don't delete the previous descriptions. Just add new version description at the end.

__version__ = "Version 0.1 # December 2017 # Initial version # Sergio Ortiz-Rojas"
__version__ = "Version 0.2 # 09/01/2018 # Adding argument for injecting wrong examples from a file # Jorge Ferrández-Tordera"
__version__ = "Version 0.3 # 18/01/2019 # Integrated training of LM and refactor to avoid code duplicity # Víctor M. Sánchez-Cartagena"
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
    groupM.add_argument('-s', '--source_lang', required=True, help="Source language")
    groupM.add_argument('-t', '--target_lang', required=True, help="Target language")
    groupM.add_argument('-d', '--source_dictionary', type=argparse.FileType('r'), required=True, help="LR gzipped probabilistic dictionary")
    groupM.add_argument('-D', '--target_dictionary', type=argparse.FileType('r'), required=True, help="RL gzipped probabilistic dictionary")
    groupM.add_argument('-f', '--source_word_freqs', type=argparse.FileType('r'), default=None, required=True, help="L language gzipped list of word frequencies")
    groupM.add_argument('-F', '--target_word_freqs', type=argparse.FileType('r'), default=None, required=True, help="R language gzipped list of word frequencies")

    groupO = parser.add_argument_group('Options')
    groupO.add_argument('-S', '--source_tokenizer_command', help="Source language tokenizer full command")
    groupO.add_argument('-T', '--target_tokenizer_command', help="Target language tokenizer full command")
    groupO.add_argument('--normalize_by_length', action='store_true', help="Normalize by length in qmax dict feature")
    groupO.add_argument('--treat_oovs', action='store_true', help="Special treatment for OOVs in qmax dict feature")
    groupO.add_argument('--qmax_limit', type=check_positive_or_zero, default=40, help="Number of max target words to be taken into account, sorted by length")
    groupO.add_argument('--disable_features_quest', action='store_false', help="Disable less important features")
    groupO.add_argument('--classifier_type', choices=['mlp', 'extra_trees', 'svm', 'nn', 'nn1', 'adaboost', 'random_forest'], default="extra_trees", help="Classifier type")
    groupO.add_argument('--dump_features', type=argparse.FileType('w'), default=None, help="Dump training features to file")
    groupO.add_argument('-b', '--block_size', type=check_positive, default=10000, help="Sentence pairs per block")
    groupO.add_argument('-p', '--processes', type=check_positive, default=max(1, cpu_count()-1), help="Number of process to use")
    groupO.add_argument('--wrong_examples_file', type=argparse.FileType('r'), default=None, help="File with wrong examples extracted to replace the synthetic examples from method used by default")
    groupO.add_argument('--features_version', type=check_positive, default=FEATURES_VERSION , help="Version of the features")
    groupO.add_argument('--disable_lang_ident', default=False, action='store_true', help="Don't apply features that use language detecting")
    groupO.add_argument('--seed', default=None, type=int, help="Seed for random number generation: by default, no seeed is used")
    groupO.add_argument('--relative_paths', action='store_true', help="Ask training to save model files by relative path if they are in the same directory as metadata. Useful if you are going to train distributable models.")

    #For LM filtering
    groupO.add_argument('--noisy_examples_file_sl', type=str, help="File with noisy text in the SL. These are used to estimate the perplexity of noisy text.")
    groupO.add_argument('--noisy_examples_file_tl', type=str, help="File with noisy text in the TL. These are used to estimate the perplexity of noisy text.")
    groupO.add_argument('--lm_dev_size', type=check_positive_or_zero, default=2000, help="Number of sentences to be removed from clean text before training LMs. These are used to estimate the perplexity of clean text.")
    groupO.add_argument('--lm_file_sl', type=str, help="SL language model output file.")
    groupO.add_argument('--lm_file_tl', type=str, help="TL language model output file.")
    groupO.add_argument('--lm_training_file_sl', type=str, help="SL text from which the SL LM is trained. If this parameter is not specified, SL LM is trained from the SL side of the input file, after removing --lm_dev_size sentences.")
    groupO.add_argument('--lm_training_file_tl', type=str, help="TL text from which the TL LM is trained. If this parameter is not specified, TL LM is trained from the TL side of the input file, after removing --lm_dev_size sentences.")
    groupO.add_argument('--lm_clean_examples_file_sl', type=str, help="File with clean text in the SL. Used to estimate the perplexity of clean text. This option must be used together with --lm_training_file_sl and both files must not have common sentences. This option replaces --lm_dev_size.")
    groupO.add_argument('--lm_clean_examples_file_tl', type=str, help="File with clean text in the TL. Used to estimate the perplexity of clean text. This option must be used together with --lm_training_file_tl and both files must not have common sentences. This option replaces --lm_dev_size.")

    groupO.add_argument('--porn_removal_train', type=argparse.FileType('r'), help="File with training dataset for FastText classifier. Each sentence must contain at the beginning the '__label__negative' or '__label__positive' according to FastText convention. It should be lowercased and tokenized.")
    groupO.add_argument('--porn_removal_test', type=argparse.FileType('r'), help="Test set to compute precision and accuracy of the porn removal classifier")
    groupO.add_argument('--porn_removal_file', type=str, help="Porn removal classifier output file")
    groupO.add_argument('--porn_removal_side', choices=['sl','tl'], default="sl", help="Whether the porn removal should be applied at the source or at the target language.")

    groupL = parser.add_argument_group('Logging')
    groupL.add_argument('-q', '--quiet', action='store_true', help='Silent logging mode')
    groupL.add_argument('--debug', action='store_true', help='Debug logging mode')
    groupL.add_argument('--logfile', type=argparse.FileType('a'), default=sys.stderr, help="Store log to a file")

    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Logging
    logging_setup(args)
    logging_level = logging.getLogger().level

    return args

# Training function: receives two file descriptors, input and test, and a
# type classifiers and trains a classifier storing it in classifier_output
# and returns some quality estimates.
def train_classifier(input_features, test_features, classifier_type, classifier_output, feat_names):
    feats=[]
    labels=[]

    # Load features and labels and format them as numpy array
    for line in input_features:
        parts = line.rstrip("\n").split("\t")
        feats.append([float(v) for v in parts[:-1]])
        labels.append(int(parts[-1]))

    dataset = dict()
    dataset['data'] = np.array(feats)
    dataset['target'] = np.array(labels)

    # Train classifier
    if classifier_type == "svm":
        clf = make_pipeline(MinMaxScaler(), svm.SVC(gamma=0.001, C=100., probability=True))
    elif classifier_type == "mlp":
        clf = MLPClassifier(verbose=True, solver='adam', alpha=1e-5, hidden_layer_sizes=(100,), random_state=1, shuffle=True, early_stopping=True, validation_fraction=0.1)
    elif classifier_type == "extra_trees":
        parameters = {
            'criterion': ('gini','entropy'),
            'n_estimators' : [100, 200, 300, 400, 500],
        }
        clf = ExtraTreesClassifier(bootstrap=True, class_weight=None,
                criterion='gini',
                max_depth=None,
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
    elif classifier_type == "nn":
        clf = neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    elif classifier_type == "nn1":
        clf = neighbors.KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    elif classifier_type == "adaboost":
        clf = AdaBoostClassifier(n_estimators=100)
    elif classifier_type == "random_forest":
        parameters = {
            'criterion': ('gini','entropy'),
            'n_estimators' : [100, 200, 300, 400, 500],
        }
        clf = RandomForestClassifier(bootstrap=True, class_weight=None,
                criterion='gini',
                max_depth=None,
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

    # If parameters is defined perform grid search
    try:
        parameters
    except NameError:
        pass
    else:
        clf = GridSearchCV(clf, parameters, n_jobs=-1)

    clf.fit(dataset['data'], dataset['target'])

    # Log sorted feature importances with their names
    if classifier_type in ('random_forest', 'adaboost', 'extra_trees'):
        if isinstance(clf, GridSearchCV):
            feat_dict = dict(zip(feat_names, clf.best_estimator_.feature_importances_))
        else:
            feat_dict = dict(zip(feat_names, clf.feature_importances_))

        logging.info("Top 10 important features: ")
        sorted_f = sorted(feat_dict.items(), key=lambda item: item[1], reverse=True)
        for feat in sorted_f[:10]:
            logging.info("\t{:.5f}: {}".format(feat[1], feat[0]))


    # Save classifier and log best params found by grid search
    if isinstance(clf, GridSearchCV):
        joblib.dump(clf.best_estimator_, classifier_output, compress=3)
        logging.info('Best classifier parameters found:')
        for k, v in clf.best_params_.items():
            logging.info('\t {}: {}'.format(k,v))
    else:
        joblib.dump(clf, classifier_output, compress=3)

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

# Writes all features of the input TUs into a temporary file
def reduce_process(output_queue, output_file):
    h = []
    last_block = 0
    while True:
        logging.debug("Reduce: heap status {}".format(h.__str__()))
        while len(h) > 0 and h[0][0] == last_block:
            nblock, filein_name = heappop(h)
            last_block += 1

            with open(filein_name, 'r') as filein:
                for i in filein:
                    output_file.write(i)
                filein.close()

            os.unlink(filein_name)

        job = output_queue.get()
        if job:
            nblock, filein_name = job
            heappush(h, (nblock, filein_name))
        else:
            logging.debug("Exiting reduce loop")
            break

    if len(h) > 0:
        logging.debug("Still elements in heap")

    while len(h) > 0 and h[0][0] == last_block:
        nblock, filein_name = heapq.heappop(h)
        last_block += 1

        with open(filein_name, 'r') as filein:
            for i in filein:
                output_file.write(i)
            filein.close()
    
        os.unlink(filein_name)

    if len(h) != 0:
        logging.error("The queue is not empty and it should!")
        sys.exit(0)

    output_file.close()

# Calculates all the features needed for the training
def worker_process(i, jobs_queue, output_queue, args):
    source_tokenizer = Tokenizer(args.source_tokenizer_command, args.source_lang)
    target_tokenizer = Tokenizer(args.target_tokenizer_command, args.target_lang)

    while True:
        job = jobs_queue.get()
        if job:
            logging.debug("Job {}".format(job.__repr__()))
            nblock, filein_name, label = job

            with open(filein_name, 'r') as filein, NamedTemporaryFile(mode="w", delete=False) as fileout:
                logging.debug("Filtering: creating temporary file {}".format(fileout.name))
                for i in filein:
                    srcsen,trgsen = i.split("\t")[:2]
                    trgsen = trgsen.strip()
                    features = feature_extract(srcsen, trgsen, source_tokenizer, target_tokenizer, args)
                    
                    for j in features:
                        fileout.write("{}".format(j))
                        fileout.write("\t")
                    fileout.write("{}".format(label))
                    fileout.write("\n")
                ojob = (nblock, fileout.name)
                fileout.close()
                filein.close()
                output_queue.put(ojob)
            os.unlink(filein_name)
        else:
            logging.debug("Exiting worker")
            source_tokenizer.close()
            target_tokenizer.close()
            break

# Divides the input among processors to speed up the throughput
def map_process(input, block_size, jobs_queue, label, first_block=0):
    nblock = first_block
    nline = 0
    mytemp = None
    for line in input:
        if (nline % block_size) == 0:
            if mytemp:
                job = (nblock, mytemp.name, label)
                mytemp.close()
                jobs_queue.put(job)
                nblock += 1
            mytemp = NamedTemporaryFile(mode="w", delete=False)
            logging.debug("Mapping: creating temporary file {}".format(mytemp.name))
        parts = line.split("\t")[0:2]
        if len(parts) >= 2:
            mytemp.write(line)
            nline += 1

    input.close()

    if nline > 0:
        job = (nblock, mytemp.name, label)
        mytemp.close()
        jobs_queue.put(job)

    return nblock

# Main loop of the program
def perform_training(args):
    time_start = default_timer()
    logging.debug("Starting process")
    logging.debug("Running {0} workers at {1} rows per block".format(args.processes, args.block_size))

    process_count = max(1, args.processes)
    maxsize = 1000 * process_count

    output_queue = Queue(maxsize = maxsize)
    worker_count = process_count

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

    # Load dictionaries
    if args.source_word_freqs:
        args.sl_word_freqs = WordZipfFreqDist(args.source_word_freqs)
    if args.target_word_freqs:
        args.tl_word_freqs = WordZipfFreqDistDoubleLinked(args.target_word_freqs)
    else:
        args.tl_word_freqs = None

    # Train porn removal classifier
    train_porn_removal(args)

    stats=None
    with open(input.name) as input_f:
        args.input=input_f
        stats=train_fluency_filter(args)
        input_f.seek(0)

        # Shuffle and get length ratio
        noisy_target_tokenizer = Tokenizer(args.target_tokenizer_command, args.target_lang)
        total_size, length_ratio, good_sentences, wrong_sentences = build_noisy_set(args.input, count_input_lines//2, count_input_lines//2, args.wrong_examples_file, args.tl_word_freqs, noisy_target_tokenizer)
        noisy_target_tokenizer.close()
    os.remove(input.name)

    args.length_ratio = length_ratio

    # Load dictionaries
    args.dict_sl_tl = ProbabilisticDictionary(args.source_dictionary)
    args.dict_tl_sl = ProbabilisticDictionary(args.target_dictionary)

    logging.info("Start computing features.")
    features_file = TemporaryFile('w+')
    # Start reducer
    reduce = Process(target = reduce_process,
                     args   = (output_queue, features_file))
    reduce.start()

    # Start workers
    jobs_queue = Queue(maxsize = maxsize)
    workers = []
    for i in range(worker_count):
        worker = Process(target = worker_process,
                         args   = (i, jobs_queue, output_queue, args))
        worker.daemon = True # dies with the parent process
        worker.start()
        workers.append(worker)


    # Mapper process (foreground - parent)
    last_block = map_process(good_sentences, args.block_size, jobs_queue, 1, 0)
    good_sentences.close()

    map_process(wrong_sentences, args.block_size, jobs_queue, 0, last_block+1)
    wrong_sentences.close()


    # Worker termination
    for _ in workers:
        jobs_queue.put(None)

    logging.info("End computing features.")

    for w in workers:
        w.join()

    # Reducer termination
    output_queue.put(None)
    reduce.join()

    features_file.seek(0)

    if args.dump_features:
        logging.info("Dumping features to " + os.path.abspath(args.dump_features.name))
        for i in features_file:
            args.dump_features.write(i)
        args.dump_features.close()
        features_file.seek(0)

    logging.info("Start training.")

    # Use 90% of the input to train and 10% for test
    if args.wrong_examples_file is not None:
        good_examples = int(count_input_lines*0.9)
        good_examples_test = int(count_input_lines*0.1)
        wrong_examples = 0
        with args.examples_file as file:
            wrong_examples = sum(1 for line in file)
        wrong_esamples_test = min(good_examples_test, int(wrong_examples*0.1))
    else:
        good_examples = int(count_input_lines//2*0.9)
        good_examples_test = int(count_input_lines//2*0.1)
        wrong_examples = good_examples
        wrong_examples_test = good_examples_test

    hgood = []
    hwrong = []
    with TemporaryFile("w+") as features_train, TemporaryFile("w+") as features_test:
        nline = 0
        for line in features_file:
            if nline < good_examples:
                features_train.write(line)
            elif nline < good_examples + good_examples_test:
                features_test.write(line)
            elif nline < good_examples + good_examples_test + wrong_examples:
                features_train.write(line)
            else:
                features_test.write(line)
            nline += 1
        
        features_train.flush()
        features_test.flush()
        
        features_train.seek(0)
        features_test.seek(0)
        hgood, hwrong = train_classifier(features_train, features_test, args.classifier_type, args.classifier, Features(None, args.disable_features_quest, args.disable_lang_ident).titles)
        features_train.close()
        features_test.close()

    logging.info("End training.")

    write_metadata(args, length_ratio, hgood, hwrong, stats)
    args.metadata.close()

    # Stats
    logging.info("Finished.")
    elapsed_time = default_timer() - time_start
    logging.info("Elapsed time {:.2f}s.".format(elapsed_time))

# Main function: setup logging and calling the main loop
def main(args):

    # Filtering
    perform_training(args)

if __name__ == '__main__':
    args = initialization()
    main(args)
