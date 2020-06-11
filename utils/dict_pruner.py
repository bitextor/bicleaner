#!/usr/bin/python
# -*- coding: utf8 -*-
import os
import re
import sys
import argparse
import logging
import traceback
import tempfile
import subprocess
#import operator
import math
import gzip
import shutil

from util import check_if_folder
from util import logging_setup
from tempfile import gettempdir

__author__ = "mbanon"
# Please, don't delete the previous descriptions. Just add new version description at the end.
__version__ = "0.1 # 20180104 # Probabilistic dictionary merger # mbanon"
__version__ = "0.2 # 20191017 # Probabilistic dictionary pruner # mbanon"


# All the scripts should have an initialization according with the usage. Template:
def initialization():
  # Getting arguments and options with argparse
  # Initialization of the argparse class
  parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
  # Mandatory parameters
  ##Dictionary file
  parser.add_argument('dictionary', type=argparse.FileType('r'), default=None, help="Dictionary file. Line format: Target Source Prob")
  ## Output file. Try to open it to check if it exists or can be created
  parser.add_argument('output', type=argparse.FileType('wb+'), default=None, help="Pruned probabilistic dictionary.")
  parser.add_argument('-v', '--version', action='version', version="%(prog)s " + __version__, help="show version of this script and exit")
  parser.add_argument('-g', '--gzipped', action='store_true', help="Compresses the output file")
  
  # Options group
  groupO = parser.add_argument_group('options')
  groupO.add_argument('-n', '--prune_ratio', type=float, default=10, help="Ratio to prune the dictionary. Translations whose probability is {} times (default) than the maximum one.".format(10))
  groupO.add_argument('-k', '--keep_tmp', action='store_true', default=False, help="This flag specifies whether removing temporal folder or not")
  groupO.add_argument('-m', '--tmp_dir', type=check_if_folder, default=gettempdir(), help="Temporary directory where creating the temporary files of this program")

  # Logging group
  groupL = parser.add_argument_group('logging')
  groupL.add_argument('-q', '--quiet', action='store_true', help='Silent logging mode')
  groupL.add_argument('--debug', action='store_true', help='Debug logging mode')
  groupL.add_argument('--logfile', type=argparse.FileType('a'), default=sys.stderr, help="Store log to a file")
  
  # Validating & parsing
  args = parser.parse_args()
  logging_setup(args)

  # Extra-checks for args here
  if (args.prune_ratio != 0):
    args.prune_ratio = math.log(args.prune_ratio)

  return args

# Prune translations
def prune_translations(ratio, source, translations, output):
    max_prob = -sys.float_info.max
    
    # For each translation, probability and log(probability)
    for t, p, l in translations:
        if l > max_prob:
            max_prob = l
    threshold = max_prob - ratio
    for t, p, l in translations:
        if l >= threshold:
            output.write(t + " " + source + " " + p + "\n")
        else:
            logging.debug("Translation {} -> {} discarded. Prob: {} - Max prob: {}".format(source, t, p, math.exp(threshold)))
            
                                      
# Prune bilingual probabilistic dictionary
# The method keeps only translations whose probability is 10 times (by default) lower than the maximum one
def prune_dictionary(dictionary, args, output):
    logging.debug("Pruning dictionaries")
    current_source = None
    current_translations = []
    for line in dictionary:  
        parts = line.strip().split(" ")        
        
        logging.debug("Current source: {0}".format(current_source))
        logging.debug("Translations: {0}".format(current_translations))
        
        if current_source != None and parts[1] != current_source:
            prune_translations(args.prune_ratio, current_source, current_translations, output)
            current_translations = []
        current_source = parts[1]
        if (float(parts[2]) != 0):
            current_translations.append((parts[0], parts[2], math.log(float(parts[2]))))
    if len(current_translations) > 0:
        prune_translations(args.prune_ratio, current_source, current_translations, output)
                                                                                                                                                                
                                                                                                                                                                
def main(args):                                                                 
    dict_file = args.dictionary
    pruned_dict = args.output
 
    temp_file = tempfile.NamedTemporaryFile(mode="wt+", delete=(not args.keep_tmp), dir=args.tmp_dir)
    afterpruning_dict = tempfile.NamedTemporaryFile(mode="wt+", delete=(not args.keep_tmp), dir=args.tmp_dir)

    
    #Sort and remove noise      
    for e in dict_file:
        parts = e.split()
        target = parts[0] #target
        source = parts[1] #source
        try:
            prob = float(parts[2])
        except:
            #logging.warning("No frequency for pair {0} - {1}".format(w2, w1))            
            continue
        #Noise removal
        if not re.match('^[-\w\']+$', source) or not re.match('^[-\w\']+$', target):
            continue
        if target == "NULL":
            continue
        temp_file.write(target.lower())
        temp_file.write(" ")
        if source == "NULL":
            temp_file.write(source)
        else:
            temp_file.write(source.lower())
        temp_file.write(" ")
        temp_file.write("{0:1.7f}\n".format(prob))
 
    temp_file.flush()
    #Sort by source
    logging.debug("Sorting")
    sort_command = "LC_ALL=C sort {0}  -k 2,2 -k 1,1 -o {0}".format(temp_file.name) # OJO, QUE ESTO ES LEGAL
    p = subprocess.Popen(sort_command, shell=True, stdout=subprocess.PIPE)
    p.wait()
    
    temp_file.seek(0)


    #Prune
    if args.prune_ratio > 0:
        logging.debug("Pruning")        
        prune_dictionary(temp_file, args, afterpruning_dict)
    else:      
        #Return dict as it is
        logging.debug("Not pruning")
        for i in temp_file:
            afterpruning_dict.write(i)
    

    afterpruning_dict.flush()
    pruned_dict_name = afterpruning_dict.name   
    
    if args.gzipped:
        logging.debug("Return gzipped")
        with open(pruned_dict_name, 'rb') as ngzd:
            with gzip.open(pruned_dict, 'wb') as gzd:
                shutil.copyfileobj(ngzd, gzd)
    else:
        logging.debug("Not gzipped")        
        with open(pruned_dict_name, 'r') as ngzd:
            with open(pruned_dict.name, 'wb') as gzd:
                shutil.copyfile(pruned_dict_name, pruned_dict.name)
    if args.keep_tmp:
        logging.info("Temp file: {0}".format(temp_file.name))                
        
if __name__ == '__main__':
    try:
        logging_setup()
        args = initialization() # Parsing parameters
        logging_setup(args)
        main(args)  # Running main program
        logging.info("Program finished")
    except Exception as ex:
        tb = traceback.format_exc()
        logging.error(tb)
        sys.exit(1)

