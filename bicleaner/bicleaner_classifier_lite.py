#!/usr/bin/env python

import os
import sys
import argparse
import logging
import traceback
import yaml
from sklearn.externals import joblib
import numpy as np

from tempfile import NamedTemporaryFile, gettempdir
from timeit import default_timer

from mosestokenizer import MosesTokenizer

#Allows to load modules while inside or outside the package
try:
    from .features import feature_extract, Features
    from .prob_dict import ProbabilisticDictionary
    from .util import no_escaping, check_positive, check_positive_or_zero, check_positive_between_zero_and_one, logging_setup
    from .bicleaner_hardrules import *
except (ImportError, SystemError):
    from features import feature_extract, Features
    from prob_dict import ProbabilisticDictionary
    from util import no_escaping, check_positive, check_positive_or_zero, check_positive_between_zero_and_one, logging_setup
    from bicleaner_hardrules import *

#import cProfile  # search for "profile" throughout the file

__author__ = "Sergio Ortiz Rojas"
__version__ = "Version 0.1 # 07/11/2018 # Initial release # Sergio Ortiz"
__version__ = "Version 0.2 # 19/11/2018 # Forcing sklearn to avoid parallelization # Marta Bañón"

nline = 0

# All the scripts should have an initialization according with the usage. Template:
def initialization():
    global nline
    nline = 0
    logging.info("Processing arguments...")
    # Getting arguments and options with argparse
    # Initialization of the argparse class
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    # Mandatory parameters
    ## Input file. Try to open it to check if it exists
    parser.add_argument('input', type=argparse.FileType('rt'), default=None, help="Tab-separated files to be classified")      
    parser.add_argument('output', nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="Output of the classification")
    parser.add_argument('metadata', type=argparse.FileType('r'), default=None, help="Training metadata (YAML file)")    

    # Options group
    groupO = parser.add_argument_group('Optional')
    
    groupO.add_argument('--tmp_dir', default=gettempdir(), help="Temporary directory where creating the temporary files of this program")
    groupO.add_argument('-d', '--discarded_tus', type=argparse.FileType('w'), default=None, help="TSV file with discarded TUs. Discarded TUs by the classifier are written in this file in TSV file.")
    groupO.add_argument('--threshold', type=check_positive_between_zero_and_one, default=0.5, help="Threshold for classifier. If accuracy histogram is present in metadata, the interval for max value will be given as a default instead the current default.")
    
    # Logging group
    groupL = parser.add_argument_group('Logging')
    groupL.add_argument('-q', '--quiet', action='store_true', help='Silent logging mode')
    groupL.add_argument('--debug', action='store_true', help='Debug logging mode')
    groupL.add_argument('--logfile', type=argparse.FileType('a'), default=sys.stderr, help="Store log to a file")
    groupL.add_argument('-v', '--version', action='version', version="%(prog)s " + __version__, help="show version of this script and exit")

    # Validating & parsing
    # Checking if metadata is specified
    args = parser.parse_args()
    logging_setup(args)
    
    try: 
        yamlpath = os.path.dirname(os.path.abspath(args.metadata.name))

        metadata_yaml = yaml.load(args.metadata)      

        args.source_lang=metadata_yaml["source_lang"]
        args.target_lang=metadata_yaml["target_lang"]
        

        try:
            args.clf=joblib.load(yamlpath + "/" + metadata_yaml["classifier"])
        except:            
            args.clf=joblib.load(metadata_yaml["classifier"])
        
#        args.clf.n_jobs = None    
        args.classifier_type=metadata_yaml["classifier_type"]


        try:
            args.dict_sl_tl = ProbabilisticDictionary(yamlpath + "/" + metadata_yaml["source_dictionary"])
        except:
            args.dict_sl_tl = ProbabilisticDictionary(metadata_yaml["source_dictionary"])                
        try:            
            args.dict_tl_sl = ProbabilisticDictionary(yamlpath+"/"+metadata_yaml["target_dictionary"])        
        except:
            args.dict_tl_sl = ProbabilisticDictionary(metadata_yaml["target_dictionary"])        
        
                
        args.normalize_by_length = metadata_yaml["normalize_by_length"]
        args.treat_oovs = metadata_yaml["treat_oovs"]
        args.qmax_limit = metadata_yaml["qmax_limit"]
        args.disable_features_quest = metadata_yaml["disable_features_quest"]
        args.good_examples = metadata_yaml["good_examples"]
        args.wrong_examples = metadata_yaml["wrong_examples"]
        args.good_test_examples = metadata_yaml["good_test_examples"]
        args.wrong_test_examples = metadata_yaml["wrong_test_examples"]
        args.length_ratio = metadata_yaml["length_ratio"]
        
        threshold = np.argmax(metadata_yaml["accuracy_histogram"])*0.1
        logging.info("Accuracy histogram: {}".format(metadata_yaml["accuracy_histogram"]))
        logging.info("Ideal threshold: {:1.1f}".format(threshold))
        metadata_yaml["threshold"] = threshold
        logging.debug("YAML")
        logging.debug(metadata_yaml)
        parser.set_defaults(**metadata_yaml)   
   
    except:
        print("Error loading metadata")
        traceback.print_exc()
        sys.exit(1)
    
    # Ensure that directory exists; if not, create it
    if not os.path.exists(args.tmp_dir):
        os.makedirs(args.tmp_dir)

    logging.debug("Arguments processed: {}".format(str(args)))
    logging.info("Arguments processed.")
    return args

#def profile_classifier_process(i, jobs_queue, output_queue,args):
#    cProfile.runctx('classifier_process(i, jobs_queue, output_queue, args)', globals(), locals(), 'profiling-{}.out'.format(i))

def classify(args):
    global nline
    batch_size = 10000
    buf_sent = []
    buf_feat = []
    
    with MosesTokenizer(args.source_lang) as source_tokenizer, MosesTokenizer(args.target_lang) as target_tokenizer:
        for i in args.input:
            nline += 1
            parts = i.split("\t")
            if len(parts) >= 4 and len(parts[2].strip()) != 0 and len(parts[3].strip()) != 0 and wrong_tu(parts[2].strip(),parts[3].strip(), args)== False:
                buf_sent.append((1, i))
                features = feature_extract(parts[2], parts[3], source_tokenizer, target_tokenizer, args)
                buf_feat.append([float(v) for v in features])
            else:
                buf_sent.append((0, i))
            
            if (nline % batch_size) == 0:
                args.clf.set_params(n_jobs = 1)
                predictions = args.clf.predict_proba(np.array(buf_feat)) if len(buf_feat) > 0 else []
                p = iter(predictions)
                
                for k, l in buf_sent:
                    if k == 1:
                        args.output.write(l.strip())
                        args.output.write("\t")
                        args.output.write(str(next(p)[1]))
                        args.output.write("\n")
                    else:
                        args.output.write(l.strip("\n"))
                        args.output.write("\t0\n")

                buf_feat = []
                buf_sent = []

        if len(buf_sent) > 0:
            predictions = args.clf.predict_proba(np.array(buf_feat)) if len(buf_feat) > 0 else []
            p = iter(predictions)
                
            for k, l in buf_sent:
                if k == 1:
                    args.output.write(l.strip())
                    args.output.write("\t")
                    args.output.write(str(next(p)[1]))
                    args.output.write("\n")
                else:
                    args.output.write(l.strip("\n"))
                    args.output.write("\t0\n")

# Filtering input texts
def perform_classification(args):
    global nline
    time_start = default_timer()
    logging.info("Starting process")
    
    classify(args)

    # Stats
    logging.info("Finished")
    elapsed_time = default_timer() - time_start
    logging.info("Total: {0} rows".format(nline))
    logging.info("Elapsed time {0:.2f} s".format(elapsed_time))
    logging.info("Troughput: {0} rows/s".format(int((nline*1.0)/elapsed_time)))

def main(args):
    logging.info("Executing main program...")
    perform_classification(args)
    logging.info("Program finished")

if __name__ == '__main__':
    try:
        logging_setup()
        args = initialization() # Parsing parameters
        main(args)  # Running main program
    except Exception as ex:
        tb = traceback.format_exc()
        logging.error(tb)
        sys.exit(1)
