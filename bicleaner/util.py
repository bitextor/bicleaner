#!/usr/bin/env python

import os
import argparse
import logging
import re
import regex
import sys

# variables used by the no_escaping function
replacements = {"&amp;":  "&",
                "&#124;": "|",
                "&lt;":   "<",
                "&gt;":   ">",
                "&apos":  "'",
                "&quot;": '"',
                "&#91;":  "[",
                "&#93;":  "]"}

substrs = sorted(replacements, key=len, reverse=True)
nrregexp = re.compile('|'.join(map(re.escape, substrs)))

regex_alpha = regex.compile("^[[:alpha:]]+$")


# Back-replacements of strings mischanged by the Moses tokenizer
def no_escaping(text):
    global nrregexp, replacements
    return nrregexp.sub(lambda match: replacements[match.group(0)], text)

# Check if the argument of a program (argparse) is positive or zero
def check_positive_between_zero_and_one(value):
    ivalue = float(value)
    if ivalue < 0 or ivalue > 1:
        raise argparse.ArgumentTypeError("%s is an invalid float value between 0 and 1" % value)
    return ivalue

# Check if the argument of a program (argparse) is positive or zero
def check_positive_or_zero(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

# Check if the argument of a program (argparse) is strictly positive
def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

# Check if the argument of a program (argparse) is strictly positive
def check_if_folder(path):
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError("%s is not a directory" % path)
    return path

# Logging config
def logging_setup(args = None):
    logger = logging.getLogger()
    logger.handlers = [] # Removing default handler to avoid duplication of log messages
    logger.setLevel(logging.ERROR)
    
    h = logging.StreamHandler(sys.stderr)
    if args != None:
       h = logging.StreamHandler(args.logfile)
      
    h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(h)

    logger.setLevel(logging.INFO)
    
    if args != None:
        if not args.quiet:
            logger.setLevel(logging.INFO)
        if args.debug:
            logger.setLevel(logging.DEBUG)


