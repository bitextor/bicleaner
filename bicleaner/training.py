import logging
import os
import random
import math
from tempfile import TemporaryFile, NamedTemporaryFile
import typing

try:
    from .lm import DualLMFluencyFilter,LMType, DualLMStats
    from .util import shuffle_file
except (SystemError, ImportError):
    from lm import DualLMFluencyFilter,LMType, DualLMStats
    from util import shuffle_file


def shuffle_lm_training_text(input: typing.TextIO,dev_size: int ) -> (str,str,str,str):

    dev_sl=NamedTemporaryFile("w",delete=False)
    dev_tl=NamedTemporaryFile("w",delete=False)
    train_sl=NamedTemporaryFile("w",delete=False)
    train_tl=NamedTemporaryFile("w",delete=False)

    with TemporaryFile("w+") as temp_sl, TemporaryFile("w+") as temp_tl, TemporaryFile("w+") as shuf_sl, TemporaryFile("w+") as shuf_tl:
        #Read tab-separated input and write its content into two different files
        for line in input:
            parts=line.rstrip("\n").split("\t")
            line_sl=parts[0]
            line_tl=parts[1]
            temp_sl.write(line_sl)
            temp_sl.write("\n")
            temp_tl.write(line_tl)
            temp_tl.write("\n")
        temp_sl.flush()
        temp_tl.flush()
        temp_sl.seek(0)
        temp_tl.seek(0)

        #Shuffle the independent files
        shuffle_file(temp_sl, shuf_sl)
        shuffle_file(temp_tl, shuf_tl)

        #read them and split between dev and train
        shuf_sl.seek(0)
        shuf_tl.seek(0)

        for i in range(dev_size):
            line=shuf_sl.readline()
            dev_sl.write(line)

            line=shuf_tl.readline()
            dev_tl.write(line)

        for line in shuf_sl:
            train_sl.write(line)

        for line in shuf_tl:
            train_tl.write(line)

    dev_sl.close()
    dev_tl.close()
    train_sl.close()
    train_tl.close()

    return train_sl.name, train_tl.name, dev_sl.name, dev_tl.name




def train_fluency_filter(args):
    # Prepare corpora:
    # Input corpora for training the classifier split in 2 parts:
    #  - Training data for LM
    #  - Validation set for estimating perplexity of clean text
    # Input noisy corpus used as validation set for estimating perplexity of noisy text

    logging.info("Training LM-based fluency filter")

    if not (args.lm_file_sl and args.lm_file_tl):
        return None

        


    inputIsTmp=True
    if args.lm_training_file_sl and args.lm_training_file_tl and args.lm_clean_examples_file_sl and args.lm_clean_examples_file_tl:
        inputIsTmp=False
        lm_train_path_sl=args.lm_training_file_sl
        lm_train_path_tl=args.lm_training_file_tl
        lm_dev_clean_sl=args.lm_clean_examples_file_sl
        lm_dev_clean_tl=args.lm_clean_examples_file_tl
        logging.info("SL LM training corpus: {}".format(lm_train_path_sl))
        logging.info("TL LM training corpus: {}".format(lm_train_path_tl))
        logging.info("SL LM dev clean corpus: {}".format(lm_dev_clean_sl))
        logging.info("TL LM dev clean corpus: {}".format(lm_dev_clean_tl))
        logging.info("SL LM dev noisy corpus: {}".format(args.noisy_examples_file_sl))
        logging.info("TL LM dev noisy corpus: {}".format(args.noisy_examples_file_tl))
    else:
        logging.info("SL & TL LM training corpora have been obtained from tab-separated input file (the same ones used for training the Random Forest classifier), after randomly removing {} sentences".format(args.lm_dev_size))
        logging.info("SL & TL LM dev clean corpora have been randomly selected from input input file (the same used for training the Random Forest classifier): {} sentences".format(args.lm_dev_size))
       
        
        lm_train_path_sl,lm_train_path_tl, lm_dev_clean_sl, lm_dev_clean_tl = shuffle_lm_training_text(args.input,args.lm_dev_size)


        if not (args.noisy_examples_file_sl):
            #build synthetic noise
            args.noisy_examples_file_sl = shuffle_chars(lm_train_path_sl)    
        logging.info("SL LM dev noisy corpus: {}".format(args.noisy_examples_file_sl))    
            
            
        if not (args.noisy_examples_file_tl):
            #build synthetic noise
            args.noisy_examples_file_tl = shuffle_chars(lm_train_path_tl)
        logging.info("TL LM dev noisy corpus: {}".format(args.noisy_examples_file_tl))

    try:
        ff=DualLMFluencyFilter(LMType.CHARACTER,args.source_lang, args.target_lang)
        stats=ff.train(lm_train_path_sl, lm_train_path_tl,lm_dev_clean_sl,lm_dev_clean_tl, args.noisy_examples_file_sl,args.noisy_examples_file_tl, args.lm_file_sl, args.lm_file_tl)
    finally:
        if inputIsTmp:
            os.remove(lm_train_path_sl)
            os.remove(lm_train_path_tl)
            os.remove(lm_dev_clean_sl)
            os.remove(lm_dev_clean_tl)
    return stats


#Randomizes sentences' characters in a file
def shuffle_chars(input_file_path):
    logging.debug("Shuffling {0} to get noisy corpus".format(input_file_path))
    noisy_file = NamedTemporaryFile("w+", delete=False)
    logging.debug("Writing noisy file to {0}".format(noisy_file.name))    
    with open (input_file_path,  "r+") as i:
        for line in i:
            s = line.strip()
            noisy_file.write(''.join(random.sample(s,len(s)))+"\n")
        
        i.flush()
        i.seek(0)
    
        noisy_file.flush()
        noisy_file.seek(0)    
    return noisy_file.name    


    

# Random shuffle corpora to ensure fairness of training and estimates.
def shuffle(input, n_aligned, n_misaligned, wrong_examples_file):
    logging.info("Shuffle starts")
    good_sentences  = TemporaryFile("w+")
    wrong_sentences = TemporaryFile("w+")
    total_size   = 0
    length_ratio = 0

    with TemporaryFile("w+") as temp:
        logging.info("Indexing file")
        # (1) Calculate the number of lines, length_ratio, offsets
        offsets = []
        nline = 0
        ssource = 0
        starget = 0
        count = 0

        for line in input:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                offsets.append(count)
                count += len(bytearray(line, "UTF-8"))
                ssource += len(parts[0])
                starget += len(parts[1])
                nline += 1
                temp.write(line)

        temp.flush()

        total_size = nline

        if total_size == 0:
            raise Exception("The input file {} is empty".format(input.name))
        elif not wrong_examples_file and  total_size < max(n_aligned, n_misaligned):
            raise Exception("Aborting... The input file {} has less lines than required by the numbers of good ({}) and wrong ({}) examples. Total lines required: {}".format(input.name, n_aligned, n_misaligned, n_aligned + n_misaligned))

        try:
            length_ratio = (ssource * 1.0)/(starget * 1.0) # It was (starget * 1.0)/(ssource * 1.0)
        except ZeroDivisionError:
            length_ratio = math.nan

        logging.info("Shuffling good sentences")
        # (2) Get good sentences
        random.shuffle(offsets)

        for i in offsets[0:n_aligned]:
            temp.seek(i)
            good_sentences.write(temp.readline())

        logging.info("Shuffling wrong sentences")
        # (3) Get wrong sentences
        if wrong_examples_file:
            # The file is already shuffled
            logging.info("Using wrong examples from file {} instead the synthetic method".format(wrong_examples_file.name))

            count = 0
            for i in wrong_examples_file:
                wrong_sentences.write(i)
                count += 1
                if count == n_misaligned:
                    break

        else:
            wrong_lines = min(total_size, n_misaligned)
            if (wrong_lines > 0):
                offsets_copy = offsets[:]
                random.shuffle(offsets)
                random.shuffle(offsets_copy)
                for i in range(wrong_lines):
                    temp.seek(offsets[i])
                    line = temp.readline()
                    parts = line.rstrip("\n").split("\t")
                    wrong_sentences.write(parts[0])
                    wrong_sentences.write("\t")
                    temp.seek(offsets_copy[i])
                    line = temp.readline()
                    parts = line.rstrip("\n").split("\t")
                    wrong_sentences.write(parts[1])
                    wrong_sentences.write("\n")
            else:
                logging.warning("Number of misaligned examples is 0")
        temp.close()
    logging.info("Shuffling ends")

    good_sentences.seek(0)
    wrong_sentences.seek(0)

    return total_size, length_ratio, good_sentences, wrong_sentences

# Calculate precision, recall and accuracy over the 0.0,1.0,0.1 histogram of
# good and  wrong alignments
def precision_recall(hgood, hwrong):
    precision = []
    recall    = []
    accuracy  = []
    total = sum(hgood) + sum(hwrong)

    for i in range(len(hgood)):
        tp = sum(hgood[i:])   # true positives
        fp = sum(hwrong[i:])  # false positives
        fn = sum(hgood[:i])   # false negatives
        tn = sum(hwrong[:i])  # true negatives
        try:
            precision.append(tp*1.0/(tp+fp))     # precision = tp/(tp+fp)
        except ZeroDivisionError:
            precision.append(math.nan)
        try:
            recall.append(tp*1.0/(tp+fn))        # recall = tp/(tp+fn)
        except ZeroDivisionError:
            recall.append(math.nan)
        try:
            accuracy.append((tp+tn)*1.0/total)   # accuracy = (tp+tn) / total
        except ZeroDivisionError:
            accuracy.append(math.nan)

    return precision, recall, accuracy


def repr_right(numeric_list, numeric_fmt = "{:1.7f}"):
    result_str = ["["]
    for i in range(len(numeric_list)):
        result_str.append(numeric_fmt.format(numeric_list[i]))
        if i < (len(numeric_list)-1):
            result_str.append(", ")
        else:
            result_str.append("]")
    return "".join(result_str)

# Write YAML with the training parameters and quality estimates
def write_metadata(myargs, length_ratio, hgood, hwrong, lm_stats:DualLMStats):
    out = myargs.metadata

    precision, recall, accuracy = precision_recall(hgood, hwrong)
    good_test_hist = "good_test_histogram: {}\n".format(hgood.__repr__())
    wrong_test_hist = "wrong_test_histogram: {}\n".format(hwrong.__repr__())
    precision_hist = "precision_histogram: {}\n".format(repr_right(precision))
    recall_hist = "recall_histogram: {}\n".format(repr_right(recall))
    accuracy_hist = "accuracy_histogram: {}\n".format(repr_right(accuracy))
    logging.debug(good_test_hist)
    logging.debug(wrong_test_hist)
    logging.debug(precision_hist)
    logging.debug(recall_hist)
    logging.debug(accuracy_hist)

    # Writing it by hand (not using YAML libraries) to preserve the order
    out.write("classifier: {}\n".format(os.path.abspath(myargs.classifier.name)))
    out.write("classifier_type: {}\n".format(myargs.classifier_type))
    out.write("source_lang: {}\n".format(myargs.source_lang))
    out.write("target_lang: {}\n".format(myargs.target_lang))
    out.write("source_dictionary: {}\n".format(os.path.abspath(myargs.source_dictionary.name)))
    out.write("target_dictionary: {}\n".format(os.path.abspath(myargs.target_dictionary.name)))
    out.write("normalize_by_length: {}\n".format(myargs.normalize_by_length))
    out.write("treat_oovs: {}\n".format(myargs.treat_oovs))
    out.write("qmax_limit: {}\n".format(myargs.qmax_limit))
    out.write("disable_features_quest: {}\n".format(myargs.disable_features_quest))
    out.write("good_examples: {}\n".format(myargs.good_examples))
    out.write("wrong_examples: {}\n".format(myargs.wrong_examples))
    out.write("good_test_examples: {}\n".format(myargs.good_test_examples))
    out.write("wrong_test_examples: {}\n".format(myargs.wrong_test_examples))
    out.write(good_test_hist)
    out.write(wrong_test_hist)
    out.write(precision_hist)
    out.write(recall_hist)
    out.write(accuracy_hist)
    out.write("length_ratio: {:1.7f}\n".format(length_ratio))
    out.write("features_version: {}\n".format(myargs.features_version))

    if lm_stats != None:
        out.write("source_lm: {}\n".format(os.path.abspath(myargs.lm_file_sl)))
        out.write("target_lm: {}\n".format(os.path.abspath(myargs.lm_file_tl)))
        out.write("lm_type: {}\n".format(str(LMType.CHARACTER)))
        out.write("clean_mean_perp: {}\n".format(lm_stats.clean_mean) )
        out.write("clean_stddev_perp: {}\n".format(lm_stats.clean_stddev) )
        out.write("noisy_mean_perp: {}\n".format(lm_stats.noisy_mean) )
        out.write("noisy_stddev_perp: {}\n".format(lm_stats.noisy_stddev) )
    out.write("disable_lang_ident: {}\n".format(myargs.disable_lang_ident))     
