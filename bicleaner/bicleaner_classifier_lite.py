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
from toolwrapper import ToolWrapper
from mosestokenizer import MosesTokenizer

#Allows to load modules while inside or outside the package
try:
    from .features import feature_extract, Features
    from .prob_dict import ProbabilisticDictionary
    from .lm import DualLMFluencyFilter,LMType, DualLMStats
    from .util import no_escaping, check_positive, check_positive_or_zero, check_positive_between_zero_and_one, logging_setup
    from .bicleaner_hardrules import *
except (ImportError, SystemError):
    from features import feature_extract, Features
    from prob_dict import ProbabilisticDictionary
    from lm import DualLMFluencyFilter,LMType, DualLMStats
    from util import no_escaping, check_positive, check_positive_or_zero, check_positive_between_zero_and_one, logging_setup
    from bicleaner_hardrules import *

#import cProfile  # search for "profile" throughout the file

__author__ = "Sergio Ortiz Rojas"
__version__ = "Version 0.1 # 07/11/2018 # Initial release # Sergio Ortiz"
__version__ = "Version 0.2 # 19/11/2018 # Forcing sklearn to avoid parallelization # Marta Bañón"
__version__ = "Version 0.3 # 17/01/2019 # Adding fluency filter # Víctor M. Sánchez-Cartagena"

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
    groupO.add_argument("-S", "--source_tokeniser_path", type=str, help="Source language (SL) tokeniser executable absolute path")
    groupO.add_argument("-T", "--target_tokeniser_path", type=str, help="Target language (TL) tokeniser executable absolute path")
    groupO.add_argument('--tmp_dir', default=gettempdir(), help="Temporary directory where creating the temporary files of this program")
    groupO.add_argument('-d', '--discarded_tus', type=argparse.FileType('w'), default=None, help="TSV file with discarded TUs. Discarded TUs by the classifier are written in this file in TSV file.")
    groupO.add_argument('--threshold', type=check_positive_between_zero_and_one, default=0.5, help="Threshold for classifier. If accuracy histogram is present in metadata, the interval for max value will be given as a default instead the current default.")
    groupO.add_argument('--lm_threshold',type=check_positive_between_zero_and_one, default=0.5, help="Threshold for language model fluency scoring. All TUs whose LM fluency score falls below the threshold will are removed (classifier score set to 0), unless the option --keep_lm_result set.")
    groupO.add_argument('--keep_lm_result',action='store_true', help="Add an additional column to the results with the language model fluency score and do not discard any TU based on that score.")
    groupO.add_argument('--score_only',action='store_true', help="Only output one column which is the bicleaner score", default=False)

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
        if "source_tokeniser_path" in metadata_yaml:
            args.source_tokeniser_path=metadata_yaml["source_tokeniser_path"]
        if "target_tokeniser_path" in metadata_yaml:
            args.target_tokeniser_path=metadata_yaml["target_tokeniser_path"]

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


        args.normalize_by_length = metadata_yaml["normalize_by_length"]
        args.treat_oovs = metadata_yaml["treat_oovs"]
        args.qmax_limit = metadata_yaml["qmax_limit"]
        args.disable_features_quest = metadata_yaml["disable_features_quest"]
        args.good_examples = metadata_yaml["good_examples"]
        args.wrong_examples = metadata_yaml["wrong_examples"]
        args.good_test_examples = metadata_yaml["good_test_examples"]
        args.wrong_test_examples = metadata_yaml["wrong_test_examples"]
        args.length_ratio = metadata_yaml["length_ratio"]
        args.features_version = 1 if "features_version" not in metadata_yaml else int(metadata_yaml["features_version"])

        threshold = np.argmax(metadata_yaml["accuracy_histogram"])*0.1
        logging.info("Accuracy histogram: {}".format(metadata_yaml["accuracy_histogram"]))
        logging.info("Ideal threshold: {:1.1f}".format(threshold))
        metadata_yaml["threshold"] = threshold

        #Load LM stuff if model was trained with it
        if "source_lm" in metadata_yaml and "target_lm" in metadata_yaml:
            lmFilter = DualLMFluencyFilter( LMType[metadata_yaml['lm_type']] ,args.source_lang, args.target_lang)
            stats=DualLMStats( metadata_yaml['clean_mean_perp'],metadata_yaml['clean_stddev_perp'],metadata_yaml['noisy_mean_perp'],metadata_yaml['noisy_stddev_perp'] )

            fullpath_source_lm=os.path.join(yamlpath,metadata_yaml['source_lm'])
            if os.path.isfile(fullpath_source_lm):
                source_lm= fullpath_source_lm
            else:
                source_lm= metadata_yaml['source_lm']

            fullpath_target_lm=os.path.join(yamlpath,metadata_yaml['target_lm'])
            if os.path.isfile(fullpath_target_lm):
                target_lm=fullpath_target_lm
            else:
                target_lm=metadata_yaml['target_lm']
            lmFilter.load(source_lm, target_lm ,stats)
            args.lm_filter=lmFilter
        else:
            args.lm_filter=None

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


    if args.score_only and args.keep_lm_result:
        raise AssertionError("Conflicting arguments, cannot output bicleaner score only AND keep language model result!")


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
    if args.source_tokeniser_path:
        source_tokeniser = ToolWrapper(args.source_tokeniser_path.split(' '))
    else:
        source_tokeniser = MosesTokenizer(args.source_lang)
    if args.target_tokeniser_path:
        target_tokeniser = ToolWrapper(args.target_tokeniser_path.split(' '))
    else:
        target_tokeniser = MosesTokenizer(args.target_lang)
    for i in args.input:
        nline += 1
        parts = i.split("\t")

        sl_sentence=None
        tl_sentence=None
        if len(parts) >= 4:
            sl_sentence=parts[2]
            tl_sentence=parts[3]

        if len(parts) == 2:
            sl_sentence=parts[0]
            tl_sentence=parts[1]

        if sl_sentence and tl_sentence and len(sl_sentence.strip()) != 0 and len(tl_sentence.strip()) != 0 and wrong_tu(sl_sentence.strip(),tl_sentence.strip(), args)== False:
            lmScore=None
            if args.lm_filter:
                lmScore=args.lm_filter.score(sl_sentence,tl_sentence)
            if lmScore != None and lmScore < args.lm_threshold and not args.keep_lm_result:
                buf_sent.append((0, i,lmScore))
            else:
                buf_sent.append((1, i,lmScore))
                features = feature_extract(sl_sentence, tl_sentence, source_tokeniser, target_tokeniser, args)
                buf_feat.append([float(v) for v in features])
        else:
            lmScore=None
            if args.lm_filter:
                lmScore=0
            buf_sent.append((0, i, lmScore))

        if (nline % batch_size) == 0:
            args.clf.set_params(n_jobs = 1)
            predictions = args.clf.predict_proba(np.array(buf_feat)) if len(buf_feat) > 0 else []
            p = iter(predictions)

            for k, l, lmScore in buf_sent:
                if k == 1:

                    if args.score_only:
                        args.output.write(str(next(p)[1]))
                    else:

                        args.output.write(l.strip())
                        args.output.write("\t")
                        args.output.write(str(next(p)[1]))
                        if lmScore != None and args.keep_lm_result:
                            args.output.write("\t")
                            args.output.write(str(lmScore))
                    args.output.write("\n")
                else:

                    if args.score_only:
                        args.output.write("0")

                    else:
                        args.output.write(l.strip("\n"))
                        args.output.write("\t0")
                        if lmScore != None and args.keep_lm_result:
                            args.output.write("\t0")
                    args.output.write("\n")

            buf_feat = []
            buf_sent = []

    if len(buf_sent) > 0:
        predictions = args.clf.predict_proba(np.array(buf_feat)) if len(buf_feat) > 0 else []
        p = iter(predictions)

        for k, l, lmScore in buf_sent:
            if k == 1:

                if args.score_only:
                    args.output.write(str(next(p)[1]))
                else:

                    args.output.write(l.strip())
                    args.output.write("\t")
                    args.output.write(str(next(p)[1]))
                    if lmScore != None and args.keep_lm_result:
                        args.output.write("\t")
                        args.output.write(str(lmScore))
                args.output.write("\n")
            else:

                if args.score_only:
                    args.output.write("0")

                else:
                    args.output.write(l.strip("\n"))
                    args.output.write("\t0")
                    if lmScore != None and args.keep_lm_result:
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
