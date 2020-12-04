#!/usr/bin/env python

import argparse
import logging
import numpy as np
import os
import random
import sys
import json
from timeit import default_timer
from tempfile import TemporaryFile, NamedTemporaryFile

#Allows to load modules while inside or outside the package
try:
    from .prob_dict import ProbabilisticDictionary
    from .word_freqs_zipf import WordZipfFreqDist
    from .word_freqs_zipf_double_linked import WordZipfFreqDistDoubleLinked
    from .util import no_escaping, check_positive, check_positive_or_zero, logging_setup
    from .training import build_noisy_set
    from .tokenizer import Tokenizer
except (SystemError, ImportError):
    from prob_dict import ProbabilisticDictionary
    from word_freqs_zipf import WordZipfFreqDist
    from word_freqs_zipf_double_linked import WordZipfFreqDistDoubleLinked
    from util import no_escaping, check_positive, check_positive_or_zero, logging_setup
    from training import build_noisy_set
    from tokenizer import Tokenizer

logging_level = 0

# Argument parsing
def initialization():

    global logging_level

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)

    parser.add_argument('input',  nargs='?', type=argparse.FileType('r'), default=sys.stdin,  help="Tab-separated bilingual input file")

    groupM = parser.add_argument_group("Mandatory")
    groupM.add_argument('-s', '--source_lang', required=True, help="Source language")
    groupM.add_argument('-t', '--target_lang', required=True, help="Target language")
    groupM.add_argument('-d', '--source_dictionary', type=argparse.FileType('r'), required=True, help="LR gzipped probabilistic dictionary")
    groupM.add_argument('-D', '--target_dictionary', type=argparse.FileType('r'), required=True, help="RL gzipped probabilistic dictionary")
    groupM.add_argument('-f', '--source_word_freqs', type=argparse.FileType('r'), default=None, required=True, help="L language gzipped list of word frequencies")
    groupM.add_argument('-F', '--target_word_freqs', type=argparse.FileType('r'), default=None, required=True, help="R language gzipped list of word frequencies")
    groupM.add_argument('--good_examples_output_file', type=argparse.FileType('w'), required=True, help="File with correct examples to be used for training")
    groupM.add_argument('--wrong_examples_output_file', type=argparse.FileType('w'), required=True, help="File with wrong examples to be used for training")

    groupO = parser.add_argument_group('Options')
    groupO.add_argument('-P', '--pretokenized_input', action='store_true', default=False, help="If flag enabled, Bicleaner expects extra fields in the TSV input that contain the tokenized version of the source and target segments.")
    groupO.add_argument('-S', '--source_tokenizer_command', help="Source language tokenizer full command")
    groupO.add_argument('-T', '--target_tokenizer_command', help="Target language tokenizer full command")
    groupO.add_argument('--seed', default=None, type=int, help="Seed for random number generation: by default, no seeed is used")
    groupO.add_argument('--noise_proportions', default="17,17,17,17,17,17", help="Six types of noise are implemented in bicleaner: adding sentences that are unrelated (not parallel), replacing some words in one of the sentences by a diffeent word with similar freqnecy, replacing words randomly in one of the sentences, shuffilng words in one of the sentences, shuffling characters in one of the sentences, and truncating one of the sentences. By default, the total amount of noise is equally divided between these six types, but percentages can be modified by providing a comma-separated list of percentages. Default: 17,17,17,17,17,17")

    groupO.add_argument('--add_lm_feature', action='store_true', help="Use LM perplexities as features instead of as an independent filter. Use the arguments --lm_file_sl, --lm_file_tl, --lm_training_file_sl and --lm_training_file_tl.")

    groupL = parser.add_argument_group('Logging')
    groupL.add_argument('-q', '--quiet', action='store_true', help='Silent logging mode')
    groupL.add_argument('--debug', action='store_true', help='Debug logging mode')
    groupL.add_argument('--logfile', type=argparse.FileType('a'), default=sys.stderr, help="Store log to a file")


    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    args.wrong_examples_file=None
    # Logging
    logging_setup(args)
    logging_level = logging.getLogger().level

    return args

# Main loop of the program
def perform_training(args):
    time_start = default_timer()
    logging.debug("Starting process")

    #Read input to a named temporary file
    #We may need to read it multiple times and that would be problematic if it is sys.stdin

    # Load dictionaries
    if args.source_word_freqs:
        #args.sl_word_freqs = WordZipfFreqDist(args.source_word_freqs)
        args.sl_word_freqs = WordZipfFreqDistDoubleLinked(args.source_word_freqs)
    if args.target_word_freqs:
        #args.tl_word_freqs = WordZipfFreqDist(args.target_word_freqs)
        args.tl_word_freqs = WordZipfFreqDistDoubleLinked(args.target_word_freqs)
    else:
        args.tl_word_freqs = None

    stats=None

    count_input_lines = 0
    input = NamedTemporaryFile(mode="w",delete=False)
    for line in args.input:
        input.write(line)
        count_input_lines = count_input_lines +1
    input.close()


    # Shuffle and get length ratio
    with open(input.name) as input_f:
        noisy_target_tokenizer = Tokenizer(args.target_tokenizer_command, args.target_lang, args.pretokenized_input)
        noisy_source_tokenizer = Tokenizer(args.source_tokenizer_command, args.source_lang, args.pretokenized_input)
    
        noise_props = [float(p)/100.0 for p in args.noise_proportions.split(",")]
        total_size, length_ratio, good_sentences, wrong_sentences = build_noisy_set(input_f, count_input_lines//2, count_input_lines//2, args.wrong_examples_file, noise_props, args.sl_word_freqs, args.tl_word_freqs, noisy_target_tokenizer, noisy_source_tokenizer, args.good_examples_output_file, args.wrong_examples_output_file)
        args.good_examples_output_file.close()
        args.wrong_examples_output_file.close()
        noisy_target_tokenizer.close()
        noisy_source_tokenizer.close()
    os.remove(input.name)

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
