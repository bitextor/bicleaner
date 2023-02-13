#!/usr/bin/env python

import os
import argparse
import logging
import re
import regex
import sys
from os import path
import typing
import random

from tempfile import TemporaryFile
from toolwrapper import ToolWrapper

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

    #logger.setLevel(logging.INFO)
    
    if args != None:
        if not args.quiet:
            logger.setLevel(logging.INFO)
        if args.debug:
            logger.setLevel(logging.DEBUG)

    logging_level = logging.getLogger().level
    if logging_level <= logging.WARNING and logging_level != logging.DEBUG:
        logging.getLogger("ToolWrapper").setLevel(logging.WARNING)

def shuffle_file(input: typing.TextIO, output: typing.TextIO):
    offsets=[]
    with TemporaryFile("w+") as temp:
        count = 0
        for line in input:
            offsets.append(count)
            count += len(bytearray(line, "UTF-8"))
            temp.write(line)
        temp.flush()
        
        random.shuffle(offsets)
        
        for offset in offsets:
            temp.seek(offset)
            output.write(temp.readline())
        
# DEPRECATED!!!
class MosesTokenizer(ToolWrapper):
    """A module for interfacing with ``tokenizer.perl`` from Moses.

    This class communicates with tokenizer.perl process via pipes. When the
    MosesTokenizer object is no longer needed, the close() method should be
    called to free system resources. The class supports the context manager
    interface. If used in a with statement, the close() method is invoked
    automatically.

    >>> tokenize = MosesTokenizer('en')
    >>> tokenize('Hello World!')
    ['Hello', 'World', '!']
    """

    def __init__(self, lang="en", old_version=False):
        self.lang = lang
        program = path.join(
            path.dirname(__file__),
            "tokenizer-" + ("v1.0" if old_version else "v1.1") + ".perl"
        )
        argv = ["perl", program, "-q", "-no-escape"  , "-l", self.lang]
        if not old_version:
            # -b = disable output buffering
            # -a = aggressive hyphen splitting
            argv.extend(["-b", "-a"])
        super().__init__(argv)

    def __str__(self):
        return "MosesTokenizer(lang=\"{lang}\")".format(lang=self.lang)

    def __call__(self, sentence):
        """Tokenizes a single sentence.

        Newline characters are not allowed in the sentence to be tokenized.
        """
        assert isinstance(sentence, str)
        sentence = sentence.rstrip("\n")
        assert "\n" not in sentence
        if not sentence:
            return []
        self.writeline(sentence)
        return self.readline().split()

