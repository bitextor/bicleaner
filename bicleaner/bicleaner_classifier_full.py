#!/usr/bin/env python

import os
import sys
import argparse
import logging
import traceback
import subprocess
import re
import yaml
import joblib
import fasttext

from heapq import heappush, heappop
from multiprocessing import Queue, Process, Value, cpu_count
from tempfile import NamedTemporaryFile
from timeit import default_timer


#Allows to load modules while inside or outside the package
try:
    from .classify import classify, argument_parser
    from .prob_dict import ProbabilisticDictionary
    from .word_freqs_zipf import WordZipfFreqDist
    from .util import check_positive, check_positive_or_zero, check_positive_between_zero_and_one, logging_setup
    from .bicleaner_hardrules import load_lm_filter
    from .tokenizer import Tokenizer
except (ImportError, SystemError):
    from classify import classify, argument_parser
    from prob_dict import ProbabilisticDictionary
    from word_freqs_zipf import WordZipfFreqDist
    from util import check_positive, check_positive_or_zero, check_positive_between_zero_and_one, logging_setup
    from bicleaner_hardrules import load_lm_filter
    from tokenizer import Tokenizer

#import cProfile  # search for "profile" throughout the file

__author__ = "Sergio Ortiz Rojas"
__version__ = "Version 0.1 # 28/12/2017 # Initial release # Sergio Ortiz"
__version__ = "Version 0.8 # 25/05/2018 # Bicleaner + Hardrules integrated # Marta Bañón"
__version__ = "Version 0.9 # 27/09/2018 # Changed input parameters for feature_extract # Marta Bañón"
__version__ = "Version 0.9.1 # 03/10/2018 # YAML is mandatory # Marta Bañón"
__version__ = "Version 0.10.4 # 17/10/2018 # Default block size is now 200 # Marta Bañón"
__version__ = "Version 0.10.8 # 18/12/2018 # Generalized tokenizer # Leopoldo Pla"
__version__ = "Version 0.11.0 # 17/01/2019 # Added fluency filter # Víctor M. Sánchez-Cartagena"
__version__ = "Version 0.12 # 29/08/2019 # # Marta Bañón"
__version__ = "Version 0.13 # 30/10/2019 # Features version 3  # Marta Bañón"

logging_level = 0

def initialization():
    global logging_level

    # Validating & parsing
    parser, groupO, _ = argument_parser()
    groupO.add_argument('-b', '--block_size', type=int, default=1000, help="Sentence pairs per block")
    groupO.add_argument('-p', '--processes', type=int, default=max(1, cpu_count()-1), help="Number of processes to use")

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
            args.clf=joblib.load( os.path.join(yamlpath , metadata_yaml["classifier"]))
        except:            
            args.clf=joblib.load(metadata_yaml["classifier"])
        
        args.clf.n_jobs = 1
        args.classifier_type=metadata_yaml["classifier_type"]


        try:
            args.dict_sl_tl = ProbabilisticDictionary( os.path.join( yamlpath, metadata_yaml["source_dictionary"]))
        except:
            args.dict_sl_tl = ProbabilisticDictionary(metadata_yaml["source_dictionary"])                
        try:            
            args.dict_tl_sl = ProbabilisticDictionary( os.path.join( yamlpath , metadata_yaml["target_dictionary"]))        
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
        args.features_version = 1 if  "features_version" not in metadata_yaml else int(metadata_yaml["features_version"])

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
                args.porn_removal = None
                logging.warning("Porn removal not present in metadata, disabling.")
            else:
                try:
                    args.porn_removal = fasttext.load_model(os.path.join(yamlpath, metadata_yaml['porn_removal_file']))
                except:
                    args.porn_removal = fasttext.load_model(args.metadata_yaml['porn_removal_file'])
        else:
            args.porn_removal = None
            logging.info("Porn removal disabled")
         
        if "disable_lang_ident" in metadata_yaml:
            args.disable_lang_ident = metadata_yaml["disable_lang_ident"]
        else:
            args.disable_lang_ident = False
            
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

def classifier_process(i, jobs_queue, output_queue, args):
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

    # If there are still jobs pending
    # grab one input file and place scores at the output queue
    while True:
        job = jobs_queue.get()
        if job:
            logging.debug("Job {0}".format(job.__repr__()))
            nblock, filein_name = job
            ojob = None
            with open(filein_name, 'r') as filein, NamedTemporaryFile(mode="w", delete=False, dir=args.tmp_dir) as fileout:
                logging.debug("Classification: creating temporary filename {0}".format(fileout.name))

                # Score sentences
                classify(args, filein, fileout, lm_filter, source_tokenizer, target_tokenizer, porn_tokenizer)

                ojob = (nblock, fileout.name)
                filein.close()
                fileout.close()

            if ojob:
                output_queue.put(ojob)

            os.unlink(filein_name)
        else:
            logging.debug("Exiting worker {}".format(job.__repr__()))
            break

def mapping_process(args, jobs_queue):
    logging.info("Start mapping")
    nblock = 0
    nline = 0
    mytemp = None
    for line in args.input:
        if (nline % args.block_size) == 0:
            logging.debug("Creating block {}".format(nblock))
            if mytemp:
                job = (nblock, mytemp.name)
                mytemp.close()
                jobs_queue.put(job)
                nblock += 1
            mytemp = NamedTemporaryFile(mode="w", delete=False, dir=args.tmp_dir)
            logging.debug("Mapping: creating temporary filename {0}".format(mytemp.name))
        mytemp.write(line)

        nline += 1

    if nline > 0:
        job = (nblock, mytemp.name)
        mytemp.close()        
        jobs_queue.put(job)

    return nline

def reduce_process(output_queue, args):
    h = []
    last_block = 0
    while True:
        logging.debug("Reduce: heap status {0}".format(h.__str__()))
        while len(h) > 0 and h[0][0] == last_block:
            nblock, filein_name = heappop(h)
            last_block += 1

            with open(filein_name, 'r') as filein:
                for i in filein:
                    args.output.write(i)

                    if args.discarded_tus:
                        args.discarded_tus.write(i)
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

        os.unlink(filein_name)

    if len(h) != 0:
        logging.error("The queue is not empty and it should!")

    logging.info("Classification finished. Output available in {}".format(args.output.name))
    args.output.close()
    if args.discarded_tus:
        logging.info("Discarded TUs are available in {}".format(args.discarded_tus.name))
        args.discarded_tus.close()

# Filtering input texts
def perform_classification(args):
    global logging_level
    
    time_start = default_timer()
    logging.debug("Starting process")
    logging.debug("Running {0} workers at {1} rows per block".format(args.processes, args.block_size))

    process_count = max(1, args.processes)
    maxsize = 1000 * process_count

    output_queue = Queue(maxsize = maxsize)
    worker_count = process_count

    # Start reducer
    logging.disable(logging.INFO)
    reduce = Process(target = reduce_process,
                     args   = (output_queue, args))
    
    reduce.start()
    logging.disable(logging.DEBUG)
    
    # Start workers
    jobs_queue = Queue(maxsize = maxsize)
    workers = []

    for i in range(worker_count):

        filter = Process(target = classifier_process, #profile_classifier_process
                         args   = (i, jobs_queue, output_queue, args))
        filter.daemon = True # dies with the parent process

        filter.start()
        workers.append(filter)


    # Mapper process (foreground - parent)
    nline = mapping_process(args, jobs_queue)
    args.input.close()

    # Worker termination
    for _ in workers:

        jobs_queue.put(None)	


    logging.info("End mapping")


    for w in workers:
        w.join()

    # Reducer termination
    

    output_queue.put(None)
    reduce.join()
    

    # Stats
    logging.info("Finished")
    elapsed_time = default_timer() - time_start
    logging.info("Total: {0} rows".format(nline))
    logging.info("Elapsed time {0:.2f} s".format(elapsed_time))
    logging.info("Troughput: {0} rows/s".format(int((nline*1.0)/elapsed_time)))
    
### END PARALLELIZATION METHODS ###

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
