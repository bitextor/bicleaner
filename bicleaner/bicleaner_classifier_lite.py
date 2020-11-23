#!/usr/bin/env python

import os
import sys
import logging
import traceback
import yaml
import joblib
import fasttext
import numpy as np

from tempfile import NamedTemporaryFile
from timeit import default_timer


#Allows to load modules while inside or outside the package
try:
    from .classify import classify, argument_parser
    from .features import feature_extract
    from .prob_dict import ProbabilisticDictionary
    from .word_freqs_zipf import WordZipfFreqDist
    from .util import check_positive, check_positive_or_zero, check_positive_between_zero_and_one, logging_setup
    from .bicleaner_hardrules import load_lm_filter
    from .tokenizer import Tokenizer
except (ImportError, SystemError):
    from classify import classify, argument_parser
    from features import feature_extract
    from prob_dict import ProbabilisticDictionary
    from word_freqs_zipf import WordZipfFreqDist
    from util import check_positive, check_positive_or_zero, check_positive_between_zero_and_one, logging_setup
    from bicleaner_hardrules import load_lm_filter
    from tokenizer import Tokenizer

logging_level = 0

# All the scripts should have an initialization according with the usage. Template:
def initialization():
    global logging_level

    # Validating & parsing arguments
    parser, groupO, _ = argument_parser()
    groupO.add_argument('-b', '--block_size', type=int, default=10000, help="Sentence pairs per block")
    args = parser.parse_args()
    logging_setup(args)
    
    logging_level = logging.getLogger().level    

            
    try: 

        metadata_yaml = yaml.safe_load(args.metadata)      
        yamlpath = os.path.dirname(os.path.abspath(args.metadata.name))
        metadata_yaml["yamlpath"] = yamlpath

       
        args.source_lang=metadata_yaml["source_lang"]
        args.target_lang=metadata_yaml["target_lang"]
        if "source_tokenizer_command" in metadata_yaml:
            args.source_tokenizer_command=metadata_yaml["source_tokenizer_command"]
        if "target_tokenizer_command" in metadata_yaml:
            args.target_tokenizer_command=metadata_yaml["target_tokenizer_command"]

        try:
            args.clf=joblib.load( os.path.join( yamlpath , metadata_yaml["classifier"]))
        except:
            args.clf=joblib.load(metadata_yaml["classifier"])

#        args.clf.n_jobs = None
        args.classifier_type=metadata_yaml["classifier_type"]

        try:
            args.dict_sl_tl = ProbabilisticDictionary( os.path.join(yamlpath , metadata_yaml["source_dictionary"]))
        except:
            args.dict_sl_tl = ProbabilisticDictionary(metadata_yaml["source_dictionary"])
        try:
            args.dict_tl_sl = ProbabilisticDictionary( os.path.join(yamlpath , metadata_yaml["target_dictionary"]))
        except:
            args.dict_tl_sl = ProbabilisticDictionary(metadata_yaml["target_dictionary"])

        try:
            args.sl_word_freqs = WordZipfFreqDist( os.path.join( yamlpath, metadata_yaml["source_word_freqs"]))
        except:
            try:
                args.sl_word_freqs = WordZipfFreqDist(metadata_yaml["source_word_freqs"])
            except:
                args.sl_word_freqs = None
        try:
            args.tl_word_freqs = WordZipfFreqDist( os.path.join( yamlpath , metadata_yaml["target_word_freqs"]))
        except:
            try:
                args.tl_word_freqs = WordZipfFreqDist(metadata_yaml["target_word_freqs"])
            except:
                args.tl_word_freqs = None

        args.normalize_by_length = metadata_yaml["normalize_by_length"]
        args.treat_oovs = metadata_yaml["treat_oovs"]
        args.qmax_limit = metadata_yaml["qmax_limit"]
        args.disable_features_quest = metadata_yaml["disable_features_quest"]
        args.length_ratio = metadata_yaml["length_ratio"]
        args.features_version = 1 if "features_version" not in metadata_yaml else int(metadata_yaml["features_version"])
        
        if "disable_lang_ident" in metadata_yaml:
            args.disable_lang_ident = metadata_yaml["disable_lang_ident"]
        else:
            args.disable_lang_ident = False     
       
        

        threshold = np.argmax(metadata_yaml["accuracy_histogram"])*0.1
        logging.info("Accuracy histogram: {}".format(metadata_yaml["accuracy_histogram"]))
        logging.info("Ideal threshold: {:1.1f}".format(threshold))
        metadata_yaml["threshold"] = threshold

        #Try loading metadata for LM filtering                  
        if not args.disable_lm_filter:
            if not ("source_lm" in metadata_yaml and "target_lm" in metadata_yaml):
                args.disable_lm_filter = True
                logging.warning("LM filter not present in metadata, disabling.")
        else:
            logging.info("LM filtering disabled")

        if not args.disable_porn_removal:
            if not ("porn_removal_file" in metadata_yaml and "porn_removal_side" in metadata_yaml):
                args.porn_removal = None
                args.disable_porn_removal = True
                logging.warning("Porn removal not present in metadata, disabling.")
            else:
                try:
                    args.porn_removal = fasttext.load_model(os.path.join(yamlpath, metadata_yaml['porn_removal_file']))
                except:
                    args.porn_removal = fasttext.load_model(args.metadata_yaml['porn_removal_file'])
        else:
            args.porn_removal = None
            logging.info("Porn removal disabled")
               
        
                
        logging.debug("YAML")
        logging.debug(metadata_yaml)
        args.metadata_yaml = metadata_yaml
        parser.set_defaults(**metadata_yaml)   
   
    except:
        logging.error("Error loading metadata")
        traceback.print_exc()
        sys.exit(1)
    
    # Ensure that directory exists; if not, create it
    if not os.path.exists(args.tmp_dir):
        os.makedirs(args.tmp_dir)


    logging.debug("Arguments processed: {}".format(str(args)))
    logging.info("Arguments processed.")
    return args

# Filtering input texts
def perform_classification(args):
    time_start = default_timer()
    logging.info("Starting process")

    # Load tokenizers and LM
    source_tokenizer = Tokenizer(args.source_tokenizer_command, args.source_lang)
    target_tokenizer = Tokenizer(args.target_tokenizer_command, args.target_lang)

    if not args.disable_lm_filter:
        lm_filter = load_lm_filter(args.source_lang, args.target_lang, args.metadata_yaml, args.source_tokenizer_command, args.target_tokenizer_command)
    else:
        lm_filter = None

    if not args.disable_porn_removal:
        if args.metadata_yaml['porn_removal_side'] == 'tl':
            porn_tokenizer = Tokenizer(args.target_tokenizer_command, args.target_lang)
        else:
            porn_tokenizer = Tokenizer(args.source_tokenizer_command, args.source_lang)
    else:
        porn_tokenizer = None
    args.clf.set_params(n_jobs = 1)

    # Score sentences
    nline = classify(args, args.input, args.output, lm_filter, source_tokenizer, target_tokenizer, porn_tokenizer)

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
