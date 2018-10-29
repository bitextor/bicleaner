#!/usr/bin/env python

import logging
import math
import pycld2
import re
import regex
import random
import string

#Allows to load modules while inside or outside the package
try:
    from .util import no_escaping, regex_alpha
except (SystemError, ImportError):
    from util import no_escaping, regex_alpha
    
from collections import Counter

re_repetition          = re.compile(r'((\w)\2{2,})')

regex_punct            = regex.compile("^[[:punct:]]+$")

regex_punctuation      = regex.compile("[[:punct:]]")
regex_arabic           = regex.compile("[\p{Arabic}]")
regex_greek            = regex.compile("[\p{Greek}]")
regex_han              = regex.compile("[\p{Han}]")
regex_hangul           = regex.compile("[\p{Hangul}]")
regex_hebrew           = regex.compile("[\p{Hebrew}]")
regex_latin            = regex.compile("[\p{Latin}]")
regex_latin_extended_a = regex.compile("[\p{InLatin_Extended-A}]")
regex_latin_extended_b = regex.compile("[\p{InLatin_Extended-B}]")
regex_latin_supplement = regex.compile("[\p{InLatin-1_Supplement}]")
regex_basic_latin      = regex.compile("[\p{InBasic_Latin}]")
regex_hiragana         = regex.compile("[\p{Hiragana}]")
regex_katakana         = regex.compile("[\p{Katakana}]")
regex_cyrillic         = regex.compile("[\p{Cyrillic}]")
regex_devanagari       = regex.compile("[\p{Devanagari}]")
regex_malayalam        = regex.compile("[\p{Malayalam}]")
regex_other            = regex.compile("[^\p{Arabic}\p{Greek}\p{Han}\p{Hangul}\p{Hebrew}\p{Latin}\p{InLatin_Extended-A}\p{InLatin_Extended-B}\p{InLatin-1_Supplement}\p{InBasic_Latin}\p{Hiragana}\p{Katakana}\p{Cyrillic}\p{Devanagari}\p{Malayalam}]")


class Features(object):
    cols = ["lang1", "lang2", "length1", "length2", 
            "npunct1", "narabic1", "ngreek1", "nhan1", 
            "nhangul1", "nhebrew1", "nlatin1", "nlatin_e_a1", 
            "nlatin_e_b1", "nlatin_sup1", "nbasic_latin1", 
            "nhiragana1", "nkatakana1", "ncyrillic1", 
            "ndevanagari1", "nmalayalam1", "nother1",
            "npunct2", "narabic2", "ngreek2", "nhan2", 
            "nhangul2", "nhebrew2", "nlatin2", "nlatin_e_a2", 
            "nlatin_e_b2", "nlatin_sup2", "nbasic_latin2", 
            "nhiragana2", "nkatakana2", "ncyrillic2", 
            "ndevanagari2", "nmalayalam2", "nother2",
            "ndif1", "freq11", "freq21", "freq31", 
            "ent1", "maxrep1", "maxword1",
            "ndif2", "freq12", "freq22", "freq32", 
            "ent2", "maxrep2", "maxword2",
            "ntok1", "ntok2",
            "poisson1", "poisson2",
            "qmax1", "cov11", "cov21",
            "qmax2", "cov12", "cov22"]       
    optional = ["avg_tok_l1","avg_tok_l2",
                     "npunct_tok1", "npunct_tok2",
                     "comma1", "period1", "semicolon1", "colon1", "doubleapos1", "quot1", "slash1",
                     "comma2", "period2", "semicolon2", "colon2", "doubleapos2", "quot2", "slash2",
                     "numeric_expr1", "numeric proportion_preserved1",
                     "numeric_expr2", "numeric_proportion_preserved2",
                     "uppercase1", "capital_proportion_preserved1",
                     "uppercase2", "capital_proportion_preserved2"]

    def __init__(self, mylist, disable_feat_quest=True):
        self.feat = mylist
        self.titles = list(Features.cols)
        if disable_feat_quest:
            self.titles += Features.optional

    def __str__(self):
        
        print("TITLES: {}".format(len(self.titles)))
        print("FEATURES: {}".format(len(self.feat)))
        result = []
        for i in range(len(self.titles)):
            result.append("{}: {}".format(self.titles[i], self.feat[i]))
        
        return "\n".join(result)  
        
# Generic logistic function
def logistic(x, center, slope, maximum=1.0):
    return maximum/(1.0+math.exp(-slope*(x-center)))

# Stirling approximation to the factorial logarithm, used to rewrite Poisson
# to an exponential of sums of logarithms to improve numerical stability.
def log_stirling(n):
    fn = float(n)
    fn = n * math.log(fn) + math.log(2 * math.pi * fn) / 2 - fn + \
         (fn ** -1) / 12 - (fn ** -3) / 360 + (fn ** -5) / 1260 - \
         (fn ** -7) / 1680 + (fn ** -9) / 1188
    return(fn)

# Likelihood of a particular sentence length ratio given the overall mean
# and the Poisson probability distribution.
def feature_length_poisson(slsentence, tlsentence, ratio):
    try:
        sllen = len(slsentence)
        tllen = len(tlsentence)
        #return  (math.exp(-sllen*ratio) * (sllen*ratio)**tllen )*1.0 / (math.factorial(tllen)*1.0) # original formula, replaced below for numerical stability
        return math.exp(-sllen*ratio + tllen*math.log(sllen*ratio) - log_stirling(tllen))
    except:
        logging.warning("Overflow when computing feature_length_poisson")
        return feature_length_poisson(["a"],["a"],1.0)

# Qmax: probability product of correspondences found (using max probabilty
# rather than alignment).
def feature_dict_qmax(slwords, tlwords, dict_stot, normalize_by_length, treat_oovs, dict_ttos, limit = 20):
    logresult = 0
    
    slwords_s_a = set()
    slwords_s_n = set()
    for i in slwords:
        if regex_alpha.match(i):
           if i in dict_stot.d:
               slwords_s_a.add(i)
        else:
           slwords_s_n.add(i)
    
    tlwords2 = list(tlwords)
    tlwords2.sort(key=len, reverse=True)
    
    if treat_oovs:
        for tlword in tlwords2[0:limit]:
            if tlword not in dict_ttos:
                pass
            else:
                t = [dict_stot.get_prob_alpha(slword, tlword) for slword in slwords_s_a]
                t.extend([dict_stot.get_prob_nonalpha(slword, tlword) for slword in slwords_s_n])
                prob = max(t, default = dict_stot.smooth)
                logresult += math.log(prob)
    else:
        for tlword in tlwords2[0:limit]:
            t = [dict_stot.get_prob_alpha(slword, tlword) for slword in slwords_s_a]
            t.extend([dict_stot.get_prob_nonalpha(slword, tlword) for slword in slwords_s_n])
            prob = max(t, default = dict_stot.smooth)
            logresult += math.log(prob)
            

    if normalize_by_length:
        logresult = float(logresult)/float(max(len(tlwords), limit))

    return math.exp(logresult)


# Coverage
def feature_dict_coverage(slwords, tlwords, dict_stot):
    tlwords_s = set(tlwords)
    slfound = [slword for slword in slwords if slword in dict_stot]
    slwithtrans = [ slword for slword in slfound if  len(set(dict_stot[slword]) & tlwords_s ) >0 ]
    return [ len(slfound)*1.0 / (1+len(slwords)) , len(slwithtrans)*1.0 /(1+len(slfound)) if len(slfound) > 0 else 1.0 ]

# Average length of tokens
# IDEAS: Poisson over this? Poisson over variances of token lengths?
def feature_avg_token_len(words):
    lens=[len(w) for w in words]
    x = sum(lens) / (1+float(len(lens)))
    return logistic(x, 5, 0.5)

# Amount of punctuation marks
def feature_num_punct_marks(words):
    return logistic(len([ w for w in words if regex_punct.match(w) ]), 15, 0.5)

# Number of punctuation marks of each type (as a separate feature)
def feature_num_punct_marks_type(words):
    rvalue=[]
    for PUNCT in [",", ".", ";", ":", "''",  '"', "/"]:
        rvalue.append(len([ word for word in words if word == PUNCT ]))
  
    return [logistic(x, 10, 0.5) for x in rvalue]

# Simplify number (years?)
def simplify_number(numstr):
    if len(numstr) == 4 and numstr[0:2] == "19":
        return numstr[2:]
    else:
        return numstr

# returns 2 values: number of numberic expressions in slsentence, and proportion of them found in tlsentence
def feature_number_preservation(slwords, tlwords):
    numbersSL=set()
    numbersTL=set()
    for tok in slwords:
        if re.match(r"(^[\d].*$|^.*[\d]$)", tok) != None:
            parts=tok.split("-")
            for tokp in parts:
                candidate = simplify_number(re.sub(r"[^\d]", "", tokp))
                if candidate != "":
                    numbersSL.add(candidate)
    for tok in tlwords:
        if re.match(r"(^[\d].*$|^.*[\d]$)", tok) != None:
            parts=tok.split("-")
            for tokp in parts:
                candidate = simplify_number(re.sub(r"[^\d]", "", tokp))
                if candidate != "":
                    numbersTL.add(candidate)
    intersection = numbersSL & numbersTL
    if len(numbersSL) > 0:
        prop=len(intersection)*1.0/(1+len(numbersSL))
    else:
        prop=1.0

    return len(numbersSL), prop

# Eval the preservation of the capitalization between source and target
def feature_capitalized_preservation(slwords, tlwords):
    tlwords_s = set(tlwords)
    slcaps= [ slword for slword in slwords if slwords[0].lower() != slwords[0] ]
    slcapsfound = [slword for slword in slcaps if slword in tlwords_s]
    return [ logistic(len(slcaps), 5, 0.7), len(slcapsfound) * 1.0/len(slcaps) if len(slcaps) > 0 else 1.0 ]

# Language detected
def feature_language(sentence, code):
    reliable = False
    bytes = 0
    details = ()
 
    try:
        reliable, bytes, details = pycld2.detect(sentence)
    except:
        sent2 = "".join(filter(lambda x: x in string.printable, sentence))
        reliable, bytes, details = pycld2.detect(sent2)
        
    if not reliable:
        return 0.0
    if details[0][1] != code:
        return 0.0
    else:
        return float(details[0][2])/100.0

# Sentence length, normalized using a logistic function
def feature_sentence_length(sentence):
    x = float(len(sentence))
    return logistic(x, 40, 0.1)

# Measure number of characters of main unicode classes
def feature_character_class_dist(sentence):
    result = []
    result.append(len(regex_punctuation.findall(sentence)))
    result.append(len(regex_arabic.findall(sentence)))
    result.append(len(regex_greek.findall(sentence)))
    result.append(len(regex_han.findall(sentence)))
    result.append(len(regex_hangul.findall(sentence)))
    result.append(len(regex_hebrew.findall(sentence)))
    result.append(len(regex_latin.findall(sentence)))
    result.append(len(regex_latin_extended_a.findall(sentence)))
    result.append(len(regex_latin_extended_b.findall(sentence)))
    result.append(len(regex_latin_supplement.findall(sentence)))
    result.append(len(regex_basic_latin.findall(sentence)))
    result.append(len(regex_hiragana.findall(sentence)))
    result.append(len(regex_katakana.findall(sentence)))
    result.append(len(regex_cyrillic.findall(sentence)))
    result.append(len(regex_devanagari.findall(sentence)))
    result.append(len(regex_malayalam.findall(sentence)))
    result.append(len(regex_other.findall(sentence)))
    
    return [logistic(x, 100, 0.4) for x in result]


def entropy(s):
    l = float(len(s))
    return -sum(map(lambda a: (a/l)*math.log2(a/l), Counter(s).values()))

def feature_character_measurements(sentence):
    res = []
    
    # Number of different characters
    res.append(logistic(len(set(sentence)), 40, 0.4))
    
    # Number of 3 most frequent characters, normalized by sentence length
    c = Counter(list(sentence)).most_common(3)
    
    while len(c) != 3:
        c.append(("", 0))
        
    res.extend([float(i[1])/(1.0 + float(len(sentence))) for i in c])
    
    # Entropy of the string
    res.append(logistic(entropy(sentence), 5, 0.2))
   
    # Max number of consecutive repetitions of the same character
    repetitions = [len(i[0]) for i in re_repetition.findall(sentence)]
    x = 0 if len(repetitions) == 0 else max(repetitions)
    res.append(logistic(x, 5, 0.7))
    
    # Length of the longest pseudo-word (blanks only)
    lengths = [len(i) for i in sentence.split(" ")]
    x = 0 if len(lengths) == 0 else max(lengths)
    res.append(logistic(x, 30, 0.4))
    
    return res
    
# Main feature function: uses program options to return a suitable set of
# features at the output
def feature_extract(srcsen, trgsen, tokenize_l, tokenize_r, args):
    length_ratio = args.length_ratio
    dict12 = args.dict_sl_tl
    dict21 = args.dict_tl_sl
    normalize_by_length = args.normalize_by_length 
    qmax_limit = args.qmax_limit
    treat_oovs = args.treat_oovs
    disable_features_quest = args.disable_features_quest
    lang1 = args.source_lang
    lang2 = args.target_lang
    
#    parts = row.strip().split("\t")

#    if len(parts) == 1:
#        parts.append("")
        
    # Sentence tokenization, with and without capital letters
    left_sentence_orig_tok  = [no_escaping(t) for t in tokenize_l(srcsen)][0:250]
    right_sentence_orig_tok = [no_escaping(t) for t in tokenize_r(trgsen)][0:250]
    left_sentence_tok =  [i.lower() for i in left_sentence_orig_tok]
    right_sentence_tok = [i.lower() for i in right_sentence_orig_tok]

    features = []
     
    features.append(feature_language(srcsen, lang1))
    features.append(feature_language(trgsen, lang2))
    features.append(feature_sentence_length(srcsen))    
    features.append(feature_sentence_length(trgsen))    
    features.extend(feature_character_class_dist(srcsen))    
    features.extend(feature_character_class_dist(trgsen))    
    features.extend(feature_character_measurements(srcsen))
    features.extend(feature_character_measurements(trgsen))
    features.append(feature_sentence_length(left_sentence_tok))
    features.append(feature_sentence_length(right_sentence_tok))
    features.append(feature_length_poisson(left_sentence_tok, right_sentence_tok, length_ratio))
    features.append(feature_length_poisson(right_sentence_tok, left_sentence_tok, 1.0/length_ratio))
    features.append(feature_dict_qmax(left_sentence_tok, right_sentence_tok, dict12, normalize_by_length, treat_oovs, dict21, qmax_limit))
    features.extend(feature_dict_coverage(left_sentence_tok, right_sentence_tok, dict12))
    features.append(feature_dict_qmax(right_sentence_tok, left_sentence_tok, dict21, normalize_by_length, treat_oovs, dict12, qmax_limit))
    features.extend(feature_dict_coverage(right_sentence_tok, left_sentence_tok, dict21))
    if disable_features_quest:
        # Average token length
        features.append(feature_avg_token_len(left_sentence_tok))
        features.append(feature_avg_token_len(right_sentence_tok))

        # Number of punctuation marks
        features.append(feature_num_punct_marks(left_sentence_tok))
        features.append(feature_num_punct_marks(right_sentence_tok))

        # Number of punctuation marks of each type: dot, comma, colon,
        # semicolon, double quotes, single quotes
        features.extend(feature_num_punct_marks_type(left_sentence_tok))
        features.extend(feature_num_punct_marks_type(right_sentence_tok))

        # Numeric expression preservation
        features.extend(feature_number_preservation(left_sentence_tok, right_sentence_tok))
        features.extend(feature_number_preservation(right_sentence_tok, left_sentence_tok))

        # Capitalized letter preservation
        features.extend(feature_capitalized_preservation(left_sentence_orig_tok, right_sentence_orig_tok))
        features.extend(feature_capitalized_preservation(left_sentence_orig_tok, right_sentence_orig_tok))
     
    return features
