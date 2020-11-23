from tempfile import gettempdir
import numpy as np
import logging
import argparse
import sys
import os

try:
    from .features import feature_extract
    from .bicleaner_hardrules import wrong_tu
    from .util import check_positive, check_positive_or_zero, check_positive_between_zero_and_one, logging_setup
except (ImportError, SystemError):
    from features import feature_extract
    from bicleaner_hardrules import wrong_tu
    from util import check_positive, check_positive_or_zero, check_positive_between_zero_and_one, logging_setup

__author__ = "Sergio Ortiz Rojas"
__version__ = "Version 0.1 # 07/11/2018 # Initial release # Sergio Ortiz"
__version__ = "Version 0.2 # 19/11/2018 # Forcing sklearn to avoid parallelization # Marta Bañón"
__version__ = "Version 0.3 # 17/01/2019 # Adding fluency filter # Víctor M. Sánchez-Cartagena"
__version__ = "Version 0.12 # 29/08/2019 # # Marta Bañón"
__version__ = "Version 0.13 # 30/10/2019 # Features version 3  # Marta Bañón"

# Create an argument parser and add all the arguments
def argument_parser():
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

    return parser, groupO, groupL


# Classify sentences from input and place them at output
# that can be either files or stdin/stdout
def classify(args, input, output, lm_filter, source_tokenizer, target_tokenizer, porn_tokenizer):
    nline = 0
    buf_sent = []
    buf_sent_sl = []
    buf_sent_tl = []
    buf_score = []

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
            logging.error("ERROR: scol ({}) or tcol ({}) indexes above column number ({}) on line {}".format(args.scol, args.tcol, len(parts), nline))

        buf_sent.append(line)

        # Buffer sentences that are not empty and pass hardrules
        if sl_sentence and tl_sentence and (args.disable_hardrules or wrong_tu(sl_sentence,tl_sentence, args, lm_filter, args.porn_removal, porn_tokenizer)== False):
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
