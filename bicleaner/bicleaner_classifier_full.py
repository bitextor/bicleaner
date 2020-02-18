#!/usr/bin/env python

import os
import sys
import argparse
import logging
import traceback
import subprocess
import math
import gzip
import re
import yaml
import sklearn
#from sklearn.externals import joblib
import joblib
import numpy as np


from heapq import heappush, heappop
from multiprocessing import Queue, Process, Value, cpu_count
from tempfile import NamedTemporaryFile, gettempdir
from timeit import default_timer
from toolwrapper import ToolWrapper
from mosestokenizer import MosesTokenizer

#Allows to load modules while inside or outside the package
try:
    from .features import feature_extract, Features
    from .prob_dict import ProbabilisticDictionary
    from .util import no_escaping, check_positive, check_positive_or_zero, check_positive_between_zero_and_one, logging_setup, check_positive
    from .bicleaner_hardrules import *

except (ImportError, SystemError):
    from features import feature_extract, Features
    from prob_dict import ProbabilisticDictionary
    from util import no_escaping, check_positive, check_positive_or_zero, check_positive_between_zero_and_one, logging_setup, check_positive
    from bicleaner_hardrules import *

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

# All the scripts should have an initialization according with the usage. Template:
def initialization():
    global logging_level
    
    logging.info("Processing arguments...")
    # Getting arguments and options with argparse
    # Initialization of the argparse class
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    # Mandatory parameters
    ## Input file. Try to open it to check if it exists
    parser.add_argument('input', type=argparse.FileType('rt'), default=None, help="Tab-separated files to be classified")      
    parser.add_argument('output', nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="Output of the classification")
    parser.add_argument('metadata', type=argparse.FileType('r'), default=None, help="Training metadata (YAML file)")    

    ## Parameters required
    #groupM = parser.add_argument_group('Mandatory')


    # Options group
    groupO = parser.add_argument_group('Optional')
    groupO.add_argument("-S", "--source_tokeniser_path", type=str, help="Source language (SL) tokeniser executable absolute path")
    groupO.add_argument("-T", "--target_tokeniser_path", type=str, help="Target language (TL) tokeniser executable absolute path")

    groupO.add_argument("--scol", default=3, type=check_positive, help ="Source sentence column (starting in 1)")
    groupO.add_argument("--tcol", default=4, type=check_positive, help ="Target sentence column (starting in 1)")    
    
    groupO.add_argument('--tmp_dir', default=gettempdir(), help="Temporary directory where creating the temporary files of this program")
    groupO.add_argument('-b', '--block_size', type=int, default=200, help="Sentence pairs per block")
    groupO.add_argument('-p', '--processes', type=int, default=max(1, cpu_count()-1), help="Number of processes to use")
    
    groupO.add_argument('-d', '--discarded_tus', type=argparse.FileType('w'), default=None, help="TSV file with discarded TUs. Discarded TUs by the classifier are written in this file in TSV file.")
    groupO.add_argument('--threshold', type=check_positive_between_zero_and_one, default=0.5, help="Threshold for classifier. If accuracy histogram is present in metadata, the interval for max value will be given as a default instead the current default.")
    groupO.add_argument('--lm_threshold',type=check_positive_between_zero_and_one, default=0.5, help="Threshold for language model fluency scoring. All TUs whose LM fluency score falls below the threshold will are removed (classifier score set to 0)")
    #groupO.add_argument('--keep_lm_result',action='store_true', help="Add an additional column to the results with the language model fluency score and do not discard any TU based on that score.")

    groupO.add_argument('--score_only',action='store_true', help="Only output one column which is the bicleaner score", default=False)   
    groupO.add_argument('--disable_hardrules',action = 'store_true', help = "Disables the bicleaner_hardrules filtering (only bicleaner_classify is applied)")
    groupO.add_argument('--disable_lm_filter', action = 'store_true', help = "Disables LM filtering")


    
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
    
    if logging_level <= logging.WARNING and logging_level != logging.DEBUG:
        #Getting rid of INFO messages when Moses processes start
        logging.getLogger("MosesTokenizer").setLevel(logging.WARNING)
        logging.getLogger("MosesSentenceSplitter").setLevel(logging.WARNING)
        logging.getLogger("MosesPunctuationNormalizer").setLevel(logging.WARNING)

            
    try: 
        metadata_yaml = yaml.safe_load(args.metadata)
        yamlpath = os.path.dirname(os.path.abspath(args.metadata.name))
        metadata_yaml["yamlpath"] = yamlpath


        args.source_lang=metadata_yaml["source_lang"]
        args.target_lang=metadata_yaml["target_lang"]
        if "source_tokeniser_path" in metadata_yaml:
            args.source_tokeniser_path=metadata_yaml["source_tokeniser_path"]
        if "target_tokeniser_path" in metadata_yaml:
            args.target_tokeniser_path=metadata_yaml["target_tokeniser_path"]        

        try:
            args.clf=joblib.load( os.path.join(yamlpath , metadata_yaml["classifier"]))
        except:            
            args.clf=joblib.load(metadata_yaml["classifier"])
        
#        args.clf.n_jobs = None    
        args.classifier_type=metadata_yaml["classifier_type"]


        try:
            args.dict_sl_tl = ProbabilisticDictionary( os.path.join( yamlpath, metadata_yaml["source_dictionary"]))
        except:
            args.dict_sl_tl = ProbabilisticDictionary(metadata_yaml["source_dictionary"])                
        try:            
            args.dict_tl_sl = ProbabilisticDictionary( os.path.join( yamlpath , metadata_yaml["target_dictionary"]))        
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
        args.features_version = 1 if  "features_version" not in metadata_yaml else int(metadata_yaml["features_version"])
        
        threshold = np.argmax(metadata_yaml["accuracy_histogram"])*0.1
        logging.info("Accuracy histogram: {}".format(metadata_yaml["accuracy_histogram"]))
        logging.info("Ideal threshold: {:1.1f}".format(threshold))
        metadata_yaml["threshold"] = threshold
        
        '''
        #Load LM stuff if model was trained with it 
        if "source_lm" in metadata_yaml and "target_lm" in metadata_yaml:
            fullpath_source_lm=os.path.join(yamlpath,metadata_yaml['source_lm'])
            if os.path.isfile(fullpath_source_lm):
                args.source_lm= fullpath_source_lm
            else:
                args.source_lm= metadata_yaml['source_lm']
            
            fullpath_target_lm=os.path.join(yamlpath,metadata_yaml['target_lm'])
            if os.path.isfile(fullpath_target_lm):
                args.target_lm=fullpath_target_lm
            else:
                args.target_lm=metadata_yaml['target_lm']
            
            
            args.lm_type=LMType[metadata_yaml['lm_type']]
            stats=DualLMStats( metadata_yaml['clean_mean_perp'],metadata_yaml['clean_stddev_perp'],metadata_yaml['noisy_mean_perp'],metadata_yaml['noisy_stddev_perp'] )
            args.lm_filter_stats=stats
        else:
            args.source_lm=None
            args.target_lm=None
            args.lm_type=None
            args.lm_filter_stats=None
        '''           

        #Try loading metadata for LM filtering                  
        if not args.disable_lm_filter:
            if not ("source_lm" in metadata_yaml and "target_lm" in metadata_yaml):
                args.disable_lm_filter = True
                logging.warning("Error loading metadata. LM filtering disabled.")
        else:
            logging.info("LM filtering disabled")


         
        if "disable_lang_ident" in metadata_yaml:
            args.disable_lang_ident = metadata_yaml["disable_lang_ident"]
        else:
            args.disable_lang_ident = False
            
        logging.debug("YAML")
        logging.debug(metadata_yaml)
        args.metadata_yaml = metadata_yaml
        parser.set_defaults(**metadata_yaml)   
   
    except:
        print("Error loading metadata")
        traceback.print_exc()
        sys.exit(1)
    
    # Ensure that directory exists; if not, create it
    if not os.path.exists(args.tmp_dir):
        os.makedirs(args.tmp_dir)

    #if args.score_only and args.keep_lm_result:
    #    raise AssertionError("Conflicting arguments: cannot output bicleaner score only AND keep language model result")
    logging.debug("Arguments processed: {}".format(str(args)))
    logging.info("Arguments processed.")
    return args

#def profile_classifier_process(i, jobs_queue, output_queue,args):
#    cProfile.runctx('classifier_process(i, jobs_queue, output_queue, args)', globals(), locals(), 'profiling-{}.out'.format(i))

def classifier_process(i, jobs_queue, output_queue, args):
    
    if args.source_tokeniser_path:    
        source_tokeniser = ToolWrapper(args.source_tokeniser_path.split(' '))
    else:
        source_tokeniser = MosesTokenizer(args.source_lang)
    if args.target_tokeniser_path:
        target_tokeniser = ToolWrapper(args.target_tokeniser_path.split(' '))
    else:
        target_tokeniser = MosesTokenizer(args.target_lang)
        
    '''
    #Load LM for fluency scoring
    lm_filter=None
    if args.source_lm and args.target_lm:

        lm_filter=DualLMFluencyFilter(args.lm_type,args.source_lang, args.target_lang)
        lm_filter.load(args.source_lm, args.target_lm,args.lm_filter_stats)
    '''

    if not args.disable_lm_filter:
        lm_filter = load_lm_filter(args.source_lang, args.target_lang, args.metadata_yaml)
    else:
        lm_filter = None
                

    while True:
        job = jobs_queue.get()
        if job:
            logging.debug("Job {0}".format(job.__repr__()))
            nblock, filein_name = job
            ojob = None
            with open(filein_name, 'r') as filein, NamedTemporaryFile(mode="w", delete=False, dir=args.tmp_dir) as fileout:
                logging.debug("Classification: creating temporary filename {0}".format(fileout.name))
                feats = []
                lm_scores=[]
                
                #Create the following arrays:
                #valid_sentences: boolean, length of input. States whether each sentence passed
                #  hard rules and lm fluency filtering
                #feats: vector of tuples, input features to the classifier, length equals number
                #  of sentences in the input that passed hard rules + lm fluency filtering
                
                valid_sentences=[]
                for i in filein:
                    parts = i.split("\t")
                    sl_sentence=None
                    tl_sentence=None
                    if len(parts) >= max(args.scol, args.tcol):
                        sl_sentence=parts[args.scol-1]
                        tl_sentence=parts[args.tcol-1]
                    else:
                        logging.error("ERROR: scol ({}) or tcol ({}) indexes above column number ({})".format(args.scol, args.tcol, len(parts)))
                        
                    if sl_sentence and tl_sentence and len(sl_sentence.strip()) != 0 and len(tl_sentence.strip()) != 0 and (args.disable_hardrules or  wrong_tu(sl_sentence.strip(),tl_sentence.strip(), args, lm_filter)== False):
                        #if disable_hardrules == 1 --> the second part (and) is always true
                        features = feature_extract(sl_sentence, tl_sentence, source_tokeniser, target_tokeniser, args)
                        feats.append([float(v) for v in features])
                        valid_sentences.append(True)
                    else:
                        valid_sentences.append(False)
                    

                predictions = args.clf.predict_proba(np.array(feats)) if len(feats) > 0 else []
                filein.seek(0)

                piter = iter(predictions)

                for i, valid_sentence in zip(filein,valid_sentences):                    
                    if valid_sentence:
                        p = next(piter)
                        if args.score_only:
                            fileout.write("{0:.3f}".format((p[1])))                        
                        else:    
                            fileout.write(i.strip())
                            fileout.write("\t")
                            fileout.write("{0:.3f}".format((p[1])))
                           
                        fileout.write("\n")
                    else:
                        if args.score_only:
                            fileout.write("0")
                        else:
                            fileout.write(i.strip("\n"))
                            fileout.write("\t0")
                        fileout.write("\n")

                ojob = (nblock, fileout.name)
                filein.close()
                fileout.close()
             
            if ojob:                    
                output_queue.put(ojob)
                
            os.unlink(filein_name)
        else:
            logging.debug("Exiting worker")
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
