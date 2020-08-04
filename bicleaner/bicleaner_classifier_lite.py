#!/usr/bin/env python

import os
import sys
import argparse
import logging
import traceback
import yaml
import joblib
import numpy as np

from tempfile import NamedTemporaryFile, gettempdir
from timeit import default_timer


#Allows to load modules while inside or outside the package
try:
    from .features import feature_extract
    from .prob_dict import ProbabilisticDictionary
    from .word_freqs_zipf import WordZipfFreqDist
    from .util import check_positive, check_positive_or_zero, check_positive_between_zero_and_one, logging_setup
    from .bicleaner_hardrules import *
    from .tokenizer import Tokenizer
except (ImportError, SystemError):
    from features import feature_extract
    from prob_dict import ProbabilisticDictionary
    from word_freqs_zipf import WordZipfFreqDist
    from util import check_positive, check_positive_or_zero, check_positive_between_zero_and_one, logging_setup
    from bicleaner_hardrules import *
    from tokenizer import Tokenizer

#import cProfile  # search for "profile" throughout the file

__author__ = "Sergio Ortiz Rojas"
__version__ = "Version 0.1 # 07/11/2018 # Initial release # Sergio Ortiz"
__version__ = "Version 0.2 # 19/11/2018 # Forcing sklearn to avoid parallelization # Marta Bañón"
__version__ = "Version 0.3 # 17/01/2019 # Adding fluency filter # Víctor M. Sánchez-Cartagena"
__version__ = "Version 0.12 # 29/08/2019 # # Marta Bañón"
__version__ = "Version 0.13 # 30/10/2019 # Features version 3  # Marta Bañón"

nline = 0
logging_level = 0

# All the scripts should have an initialization according with the usage. Template:
def initialization():
    global nline
    global logging_level
    
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
    groupO.add_argument("-S", "--source_tokenizer_command", type=str, help="Source language (SL) tokenizer full command")
    groupO.add_argument("-T", "--target_tokenizer_command", type=str, help="Target language (TL) tokenizer full command")

    groupO.add_argument("--scol", default=3, type=check_positive, help ="Source sentence column (starting in 1)")
    groupO.add_argument("--tcol", default=4, type=check_positive, help ="Target sentence column (starting in 1)")    


    groupO.add_argument('--tmp_dir', default=gettempdir(), help="Temporary directory where creating the temporary files of this program")
    groupO.add_argument('-d', '--discarded_tus', type=argparse.FileType('w'), default=None, help="TSV file with discarded TUs. Discarded TUs by the classifier are written in this file in TSV file.")
    groupO.add_argument('--lm_threshold',type=check_positive_between_zero_and_one, default=0.5, help="Threshold for language model fluency scoring. All TUs whose LM fluency score falls below the threshold will are removed (classifier score set to 0), unless the option --keep_lm_result set.")
    #groupO.add_argument('--keep_lm_result',action='store_true', help="Add an additional column to the results with the language model fluency score and do not discard any TU based on that score.")
     
    groupO.add_argument('--score_only',action='store_true', help="Only output one column which is the bicleaner score", default=False)
     
    groupO.add_argument('--disable_hardrules',action = 'store_true', help = "Disables the bicleaner_hardrules filtering (only bicleaner_classify is applied)")
    groupO.add_argument('--disable_lm_filter', action = 'store_true', help = "Disables LM filtering")
    groupO.add_argument('--disable_porn_removal', default=False, action='store_true', help="Don't apply porn removal")
    groupO.add_argument('--disable_minimal_length', default=False, action='store_true', help="Don't apply minimal length rule")

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
                args.disable_porn_removal = True
                logging.warning("Porn removal not present in metadata, disabling.")
            else:
                try:
                    args.porn_removal = fasttext.load_model(os.path.join(yamlpath, metadata_yaml['porn_removal_file']))
                except:
                    args.porn_removal = fasttext.load_model(args.metadata_yaml['porn_removal_file'])
        else:
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

#def profile_classifier_process(i, jobs_queue, output_queue,args):
#    cProfile.runctx('classifier_process(i, jobs_queue, output_queue, args)', globals(), locals(), 'profiling-{}.out'.format(i))

def classify(args):
    global nline
    batch_size = 10000
    buf_sent = []
    buf_feat = []
    
    source_tokenizer = Tokenizer(args.source_tokenizer_command, args.source_lang)
    target_tokenizer = Tokenizer(args.target_tokenizer_command, args.target_lang)
    
    if not args.disable_lm_filter:
        lm_filter = load_lm_filter(args.source_lang, args.target_lang, args.metadata_yaml, args.source_tokenizer_command, args.target_tokenizer_command)
    else:
        lm_filter = None

    if not args.disable_porn_removal:
        porn_removal = args.porn_removal
        if args.metadata_yaml['porn_removal_side'] == 'tl':
            porn_tokenizer = Tokenizer(args.target_tokenizer_command, args.target_lang)
        else:
            porn_tokenizer = Tokenizer(args.source_tokenizer_command, args.source_lang)
    else:
        porn_removal = None
        porn_tokenizer = None

    for i in args.input:
        nline += 1
        parts = i.split("\t")
        
        sl_sentence=None
        tl_sentence=None
        if len(parts) >= max(args.scol, args.tcol):
            sl_sentence=parts[args.scol -1]
            tl_sentence=parts[args.tcol -1]
        else:
            logging.error("ERROR: scol ({}) or tcol ({}) indexes above column number ({}) on line {}".format(args.scol, args.tcol, len(parts), nline))
                       
        if sl_sentence and tl_sentence and len(sl_sentence.strip()) != 0 and len(tl_sentence.strip()) != 0 and (args.disable_hardrules or wrong_tu(sl_sentence.strip(),tl_sentence.strip(), args, lm_filter, porn_removal, porn_tokenizer)== False):
            buf_sent.append((1, i))
            features = feature_extract(sl_sentence, tl_sentence, source_tokenizer, target_tokenizer, args)
            buf_feat.append([float(v) for v in features])
        else:
            buf_sent.append((0, i))
        
        if (nline % batch_size) == 0:
            args.clf.set_params(n_jobs = 1)
            predictions = args.clf.predict_proba(np.array(buf_feat)) if len(buf_feat) > 0 else []
            p = iter(predictions)
                
            for k, l in buf_sent:
                if k == 1:
                    if args.score_only:
                        args.output.write("{0:.3f}".format((next(p)[1])))
                    else:
                        args.output.write(l.strip())
                        args.output.write("\t{0:.3f}".format((next(p)[1])))                       
                    args.output.write("\n")
                else:
                    if args.score_only:
                        args.output.write("0")
                    else:    
                        args.output.write(l.strip("\n"))
                        args.output.write("\t0")
                    args.output.write("\n")

            buf_feat = []
            buf_sent = []

    if len(buf_sent) > 0:
        predictions = args.clf.predict_proba(np.array(buf_feat)) if len(buf_feat) > 0 else []
        p = iter(predictions)
            
        for k, l in buf_sent:
            if k == 1:
                if args.score_only:
                    args.output.write("{0:.3f}".format((next(p)[1])))
                else:
                    args.output.write(l.strip())
                    args.output.write("\t")
                    args.output.write("{0:.3f}".format((next(p)[1])))
                args.output.write("\n")
            else:
                if args.score_only:
                    args.output.write("0")
                else:    
                    args.output.write(l.strip("\n"))
                    args.output.write("\t0")
                args.output.write("\n")
                
                
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
