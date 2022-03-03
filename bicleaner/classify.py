from tempfile import gettempdir
import numpy as np
import traceback
import argparse
import fasttext
import logging
import joblib
import yaml
import sys
import os

#Allows to load modules while inside or outside the package
try:
    from .features import feature_extract
    from .prob_dict import ProbabilisticDictionary
    from .word_freqs_zipf import WordZipfFreqDist
    from .util import check_positive, check_positive_or_zero, check_positive_between_zero_and_one, logging_setup
except (ImportError, SystemError):
    from features import feature_extract
    from prob_dict import ProbabilisticDictionary
    from word_freqs_zipf import WordZipfFreqDist
    from util import check_positive, check_positive_or_zero, check_positive_between_zero_and_one, logging_setup

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
__version__ = "Version 0.14 # 04/08/2020 # Features version 4 # Miquel Esplà, Jaume Zaragoza and Marta Bañón"
__version__ = "Version 0.15 # 02/03/2022 # # Miquel Esplà, Jaume Zaragoza and Marta Bañón"
__version__ = "Version 0.15.1 # 03/03/2022 # # Miquel Esplà, Jaume Zaragoza and Marta Bañón"


# Create an argument parser and add all the arguments
def argument_parser():
    header = "--header" in sys.argv

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

    groupO.add_argument("--header", action='store_true', help="Input file will be expected to have a header, and the output will have a header as well")
    groupO.add_argument("--scol", default=3 if not header else "src_text", type=check_positive if not header else str, help ="Source sentence column (starting in 1). The name of the field is expected instead of the position if --header is set")
    groupO.add_argument("--tcol", default=4 if not header else "trg_text", type=check_positive if not header else str, help ="Target sentence column (starting in 1). The name of the field is expected instead of the position if --header is set")


    groupO.add_argument('--tmp_dir', default=gettempdir(), help="Temporary directory where creating the temporary files of this program")
    groupO.add_argument('-d', '--discarded_tus', type=argparse.FileType('w'), default=None, help="TSV file with discarded TUs. Discarded TUs by the classifier are written in this file in TSV file.")
    groupO.add_argument('--lm_threshold',type=check_positive_between_zero_and_one, default=0.5, help="Threshold for language model fluency scoring. All TUs whose LM fluency score falls below the threshold will are removed (classifier score set to 0), unless the option --keep_lm_result set.")
    #groupO.add_argument('--keep_lm_result',action='store_true', help="Add an additional column to the results with the language model fluency score and do not discard any TU based on that score.")
     
    groupO.add_argument('--score_only',action='store_true', help="Only output one column which is the bicleaner score", default=False)
     
    groupO.add_argument('--disable_hardrules',action = 'store_true', help = "Disables the bicleaner_hardrules filtering (only bicleaner_classify is applied)")
    groupO.add_argument('--disable_lm_filter', action = 'store_true', help = "Disables LM filtering")
    groupO.add_argument('--disable_porn_removal', default=False, action='store_true', help="Don't apply porn removal")
    groupO.add_argument('--disable_minimal_length', default=False, action='store_true', help="Don't apply minimal length rule")
    groupO.add_argument('--run_all_rules', default=False, action='store_true', help="Run all rules of Hardrules instead of stopping at first discard")
    groupO.add_argument('--rules_config', type=argparse.FileType('r'), default=None, help="Hardrules configuration file")

    # Logging group
    groupL = parser.add_argument_group('Logging')
    groupL.add_argument('-q', '--quiet', action='store_true', help='Silent logging mode')
    groupL.add_argument('--debug', action='store_true', help='Debug logging mode')
    groupL.add_argument('--logfile', type=argparse.FileType('a'), default=sys.stderr, help="Store log to a file")
    groupL.add_argument('-v', '--version', action='version', version="%(prog)s " + __version__, help="show version of this script and exit")

    return parser, groupO, groupL


# Load metadata, classifier, dictionaries, wordfreqs, lm_filter and porn_removal
def load_metadata(args, parser):
    try:
        # Load YAML
        metadata_yaml = yaml.safe_load(args.metadata)
        yamlpath = os.path.dirname(os.path.abspath(args.metadata.name))
        metadata_yaml["yamlpath"] = yamlpath

        # Read language pair and tokenizers
        args.source_lang=metadata_yaml["source_lang"]
        args.target_lang=metadata_yaml["target_lang"]
        if "source_tokenizer_command" in metadata_yaml:
            args.source_tokenizer_command=metadata_yaml["source_tokenizer_command"]
        if "target_tokenizer_command" in metadata_yaml:
            args.target_tokenizer_command=metadata_yaml["target_tokenizer_command"]

        # Load classifier
        try:
            args.clf=joblib.load( os.path.join( yamlpath , metadata_yaml["classifier"]))
        except:
            args.clf=joblib.load(metadata_yaml["classifier"])
        args.clf.n_jobs = 1
        args.classifier_type=metadata_yaml["classifier_type"]

        # Load probabilistic dictionaries
        try:
            args.dict_sl_tl = ProbabilisticDictionary( os.path.join(yamlpath , metadata_yaml["source_dictionary"]))
        except:
            args.dict_sl_tl = ProbabilisticDictionary(metadata_yaml["source_dictionary"])
        try:
            args.dict_tl_sl = ProbabilisticDictionary( os.path.join(yamlpath , metadata_yaml["target_dictionary"]))
        except:
            args.dict_tl_sl = ProbabilisticDictionary(metadata_yaml["target_dictionary"])

        # Load wordfreqs
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

        # Load feature parameters
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

        # Read accuracy histogram
        threshold = np.argmax(metadata_yaml["accuracy_histogram"])*0.1
        logging.info("Accuracy histogram: {}".format(metadata_yaml["accuracy_histogram"]))
        logging.info("Ideal threshold: {:1.1f}".format(threshold))
        metadata_yaml["threshold"] = threshold

        # Try loading metadata for LM filtering
        if not args.disable_lm_filter:
            if not ("source_lm" in metadata_yaml and "target_lm" in metadata_yaml):
                args.disable_lm_filter = True
                logging.warning("LM filter not present in metadata, disabling.")
        else:
            logging.info("LM filtering disabled")

        # Try loading porn_removal model
        if not args.disable_hardrules and not args.disable_porn_removal:
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


# Classify sentences from input and place them at output
# that can be either files or stdin/stdout
def classify(args, input, output, source_tokenizer, target_tokenizer, hardrules):
    nline = 0
    buf_sent = []
    buf_sent_sl = []
    buf_sent_tl = []
    buf_score = []

    if args.header:
        args.header = False # We only need to execute the following code once
        header = next(args.input).strip().split("\t")

        # Transform fields to idxs
        if args.scol not in header:
            raise Exception(f"The provided --scol '{args.scol}' is not in the input header")
        if args.tcol not in header:
            raise Exception(f"The provided --tcol '{args.tcol}' is not in the input header")

        args.scol = int(header.index(args.scol)) + 1
        args.tcol = int(header.index(args.tcol)) + 1
        logging.info(args.scol)
        logging.info(args.tcol)

        output_header = header

        if args.score_only:
            output_header = ["bicleaner_score"]
        else:
            output_header.append("bicleaner_score")

        # Write the output header once
        args.output.write('\t'.join(output_header) + '\n')

    # Read from input file/stdin
    for line in input:
        nline += 1
        parts = line.split("\t")

        # Parse fields and buffer sentences
        sl_sentence=None
        tl_sentence=None
        if len(parts) >= max(args.scol, args.tcol):
            sl_sentence=parts[args.scol -1].strip()
            tl_sentence=parts[args.tcol -1].strip()
        else:
            logging.error("scol ({}) or tcol ({}) indexes above column number ({}) on line {}".format(args.scol, args.tcol, len(parts), nline))

        buf_sent.append(line)

        # Buffer sentences that are not empty and pass hardrules
        if sl_sentence and tl_sentence and (args.disable_hardrules or hardrules.wrong_tu(sl_sentence, tl_sentence) == False):
            buf_score.append(1)
            buf_sent_sl.append(sl_sentence)
            buf_sent_tl.append(tl_sentence)
        else:
            buf_score.append(0)

        # Score batch and empty buffers
        if (nline % args.block_size) == 0:
            classify_batch(args, output, buf_sent, buf_sent_sl, buf_sent_tl, buf_score, source_tokenizer, target_tokenizer)
            buf_sent = []
            buf_sent_sl = []
            buf_sent_tl = []
            buf_score = []

    # Score remaining sentences
    if len(buf_sent) > 0:
        classify_batch(args, output, buf_sent, buf_sent_sl, buf_sent_tl, buf_score, source_tokenizer, target_tokenizer)

    return nline

# Score a batch of sentences
def classify_batch(args, output, buf_sent, buf_sent_sl, buf_sent_tl, buf_score, source_tokenizer, target_tokenizer):
    # Tokenize
    buf_tok_sl = source_tokenizer.tokenize(buf_sent_sl)
    buf_tok_tl = target_tokenizer.tokenize(buf_sent_tl)

    # Compute features
    buf_feat = []
    for sl_sent, tl_sent, sl_sent_t, tl_sent_t in zip(buf_sent_sl, buf_sent_tl, buf_tok_sl, buf_tok_tl):
        features = feature_extract(sl_sent, tl_sent, sl_sent_t, tl_sent_t, args)
        buf_feat.append([float(v) for v in features])

    # Classifier predictions
    predictions = args.clf.predict_proba(np.array(buf_feat)) if len(buf_feat) > 0 else []
    p = iter(predictions)

    # Print sentences and scores to output
    for score, sent in zip(buf_score, buf_sent):
        if score == 1:
            if args.score_only:
                output.write("{0:.3f}".format((next(p)[1])))
            else:
                output.write(sent.strip())
                output.write("\t")
                output.write("{0:.3f}".format((next(p)[1])))
            output.write("\n")
        else:
            if args.score_only:
                output.write("0")
            else:
                output.write(sent.rstrip("\n"))
                output.write("\t0")
            output.write("\n")
