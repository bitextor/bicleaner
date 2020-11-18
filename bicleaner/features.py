#!/usr/bin/env python

import logging
import math
import pycld2
import re
import regex
import random
import string
try:
    from .lm import LMFluencyFilter,DualLMFluencyFilter,LMType, DualLMStats
except (SystemError, ImportError):
    from lm import LMFluencyFilter,DualLMFluencyFilter,LMType, DualLMStats

FEATURES_VERSION = 4

#Allows to load modules while inside or outside the package
try:
    from .util import regex_alpha
except (SystemError, ImportError):
    from util import  regex_alpha
    
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

class SegmentPair(object):
    def __init__(self, sseg, tseg, stokenizer, ttokenizer):
        self.source = sseg
        self.target = tseg

        lt = stokenizer.tokenize(sseg)
        rt = ttokenizer.tokenize(tseg)
    
        left_sentence_orig_tok  = lt[0:250]
        right_sentence_orig_tok = rt[0:250]
    
        self.source_tok =  [i.lower() for i in left_sentence_orig_tok]
        self.target_tok = [i.lower() for i in right_sentence_orig_tok]

class FeatureSet(object):
        
    def run(self):
        pass

    # Generic logistic function
    @staticmethod
    def logistic(x, center, slope, maximum=1.0):
        return maximum/(1.0+math.exp(-slope*(x-center)))


class FeatureLM(FeatureSet):
    captions = ['sl_perplexity','tl_perplexity',
                'sl_tl_perplexity_ratio','tl_sl_perplexity_ratio']

    def __init__(self, source_lang, target_lang, source_tokenizer_command, target_tokenizer_command, lm_file_sl, lm_file_tl):
        FeatureSet.__init__(self)
        self.sl_lm = LMFluencyFilter(LMType.CHARACTER, source_lang, source_tokenizer_command)
        self.sl_lm.load_lm(lm_file_sl)
        self.tl_lm = LMFluencyFilter(LMType.CHARACTER, target_lang, target_tokenizer_command)
        self.tl_lm.load_lm(lm_file_tl)

    def run(self, segment_pair):
        features = []
        sl_score = self.sl_lm.score(segment_pair.source, True)
        features.append(sl_score)
        
        tl_score = self.tl_lm.score(segment_pair.target, True)
        features.append(tl_score)

        features.append(tl_score/sl_score)
        features.append(sl_score/tl_score)
        return features


class FeatureCapitalizedPreservation(FeatureSet):
    captions = ["uppercase1", "capital_proportion_preserved1",
                "uppercase2", "capital_proportion_preserved2"]

    def __init__(self):
        FeatureSet.__init__(self)

    def run(self, segment_pair):
        features = []
        features.extend(FeatureCapitalizedPreservation.feature_capitalized_preservation(segment_pair.source_tok, segment_pair.target_tok))
        features.extend(FeatureCapitalizedPreservation.feature_capitalized_preservation(segment_pair.target_tok, segment_pair.source_tok))
        return features

    # Eval the preservation of the capitalization between source and target
    @staticmethod
    def feature_capitalized_preservation(slwords, tlwords):
        tlwords_s = set(tlwords)
        slcaps= [ slword for slword in slwords if slwords[0].lower() != slwords[0] ]
        slcapsfound = [slword for slword in slcaps if slword in tlwords_s]
        return [ FeatureSet.logistic(len(slcaps), 5, 0.7), len(slcapsfound) * 1.0/len(slcaps) if len(slcaps) > 0 else 1.0 ]


class FeatureNumberPreservation(FeatureSet):
    captions = ["numeric_expr1", "numeric_proportion_preserved1",
                "numeric_expr2", "numeric_proportion_preserved2"]

    def __init__(self):
        FeatureSet.__init__(self)

    def run(self, segment_pair):
        features = []
        features.extend(FeatureNumberPreservation.feature_number_preservation(segment_pair.source, segment_pair.target_tok))
        features.extend(FeatureNumberPreservation.feature_number_preservation(segment_pair.target, segment_pair.source_tok))
        return features

    # returns 2 values: number of numberic expressions in slsentence, and proportion of them found in tlsentence
    @staticmethod
    def feature_number_preservation(slwords, tlwords):
        numbersSL=set()
        numbersTL=set()
        for tok in slwords:
            if re.match(r"(^[\d].*$|^.*[\d]$)", tok) != None:
                parts=tok.split("-")
                for tokp in parts:
                    candidate = FeatureNumberPreservation.simplify_number(re.sub(r"[^\d]", "", tokp))
                    if candidate != "":
                        numbersSL.add(candidate)
        for tok in tlwords:
            if re.match(r"(^[\d].*$|^.*[\d]$)", tok) != None:
                parts=tok.split("-")
                for tokp in parts:
                    candidate = FeatureNumberPreservation.simplify_number(re.sub(r"[^\d]", "", tokp))
                    if candidate != "":
                        numbersTL.add(candidate)
        intersection = numbersSL & numbersTL
        if len(numbersSL) > 0:
            prop=len(intersection)*1.0/(1+len(numbersSL))
        else:
            prop=1.0
    
        return len(numbersSL), prop


    # Simplify number (years?)
    @staticmethod
    def simplify_number(numstr):
        if len(numstr) == 4 and numstr[0:2] == "19":
            return numstr[2:]
        else:
            return numstr


class FeatureNumPunctMarksType(FeatureSet):
    captions = ["comma1", "period1", "semicolon1", "colon1", "doubleapos1", "quot1", "slash1",
                "comma2", "period2", "semicolon2", "colon2", "doubleapos2", "quot2", "slash2"]

    def __init__(self):
        FeatureSet.__init__(self)

    def run(self, segment_pair):
        features = []
        features.extend(FeatureNumPunctMarksType.feature_num_punct_marks_type(segment_pair.source_tok))
        features.extend(FeatureNumPunctMarksType.feature_num_punct_marks_type(segment_pair.target_tok))
        return features

    # Number of punctuation marks of each type (as a separate feature)
    @staticmethod
    def feature_num_punct_marks_type(words):
        rvalue=[]
        for PUNCT in [",", ".", ";", ":", "''",  '"', "/"]:
            rvalue.append(len([ word for word in words if word == PUNCT ]))
    
        return [FeatureSet.logistic(x, 10, 0.5) for x in rvalue]


class FeatureNumPunctMarks(FeatureSet):
    captions = ["npunct_tok1", "npunct_tok2"]

    def __init__(self):
        FeatureSet.__init__(self)

    def run(self, segment_pair):
        features = []
        features.append(FeatureNumPunctMarks.feature_num_punct_marks(segment_pair.source_tok))
        features.append(FeatureNumPunctMarks.feature_num_punct_marks(segment_pair.target_tok))
        return features

    # Amount of punctuation marks
    @staticmethod
    def feature_num_punct_marks(words):
        return FeatureSet.logistic(len([ w for w in words if regex_punct.match(w) ]), 15, 0.5)

class FeatureAvgWordLneght(FeatureSet):
    captions = ["avg_tok_l1","avg_tok_l2"]

    def __init__(self):
        FeatureSet.__init__(self)

    def run(self, segment_pair):
        features = []
        features.append(FeatureAvgWordLneght.feature_avg_token_len(segment_pair.source_tok))
        features.append(FeatureAvgWordLneght.feature_avg_token_len(segment_pair.target_tok))
        return features

    # Average length of tokens
    # IDEAS: Poisson over this? Poisson over variances of token lengths?
    @staticmethod
    def feature_avg_token_len(words):
        lens=[len(w) for w in words]
        x = sum(lens) / (1+float(len(lens)))
        return FeatureSet.logistic(x, 5, 0.5)

class FeautreDicCoverageQuartiles(FeatureSet):
    captions = ["cov2q1", "cov2transq1", "cov2q2", "cov2transq2", "cov2q3", "cov2transq3", "cov2q4", "cov2transq4",
                "cov1q1", "cov1transq1", "cov1q2", "cov1transq2", "cov1q3", "cov1transq3", "cov1q4", "cov1transq4"]

    def __init__(self, dict_stot, dict_ttos, l1freqs, l2freqs):
        FeatureSet.__init__(self)
        self.dict_stot = dict_stot
        self.dict_ttos = dict_ttos
        self.l1freqs = l1freqs
        self.l2freqs = l2freqs

    def run(self, segment_pair):
        features = []
        features.extend(FeautreDicCoverageQuartiles.feature_dict_coverage_zipf_freq(segment_pair.source_tok, segment_pair.target_tok, self.dict_stot, self.l2freqs))
        features.extend(FeautreDicCoverageQuartiles.feature_dict_coverage_zipf_freq(segment_pair.target_tok, segment_pair.source_tok, self.dict_ttos, self.l1freqs))
        return features

    @staticmethod
    def feature_dict_coverage_zipf_freq(slwords, tlwords, dict_stot, freqs):
        t_word_splits = freqs.split_sentence_by_freq(tlwords)
        output = []
        for i in range(0, 4):
            output.extend(FeautreDicCoverage.feature_dict_coverage(slwords, t_word_splits[i], dict_stot))
        return output

class FeautreDicCoverage(FeatureSet):
    captions = ["cov2", "cov2trans", "cov1", "cov1trans"]

    def __init__(self, dict_stot, dict_ttos):
        FeatureSet.__init__(self)
        self.dict_stot = dict_stot
        self.dict_ttos = dict_ttos

    def run(self, segment_pair):
        features = []
        features.extend(FeautreDicCoverage.feature_dict_coverage(segment_pair.source_tok, segment_pair.target_tok, self.dict_stot))
        features.extend(FeautreDicCoverage.feature_dict_coverage(segment_pair.target_tok, segment_pair.source_tok, self.dict_ttos))
        return features

    # Coverage
    @staticmethod
    def feature_dict_coverage(slwords, tlwords, dict_stot):
        slwords_s = set(slwords)
        tlfound = [tlword for tlword in tlwords if tlword in dict_stot.dinv]
        tlwithtrans = [ tlword for tlword in tlfound if  len(dict_stot.dinv[tlword] & slwords_s ) >0 ]
        return [ len(tlfound)*1.0 / (1+len(tlwords)) , len(tlwithtrans)*1.0 /(1+len(tlfound)) if len(tlfound) > 0 else 1.0 ]

class FeatureQmax(FeatureSet):
    captions = ["qmax_freq1q1", "qmax_freq1q2", "qmax_freq1q3", "qmax_freq1q4",
                "qmax_freq2q1", "qmax_freq2q2", "qmax_freq2q3", "qmax_freq2q4"]

    def __init__(self, dict_stot, dict_ttos, normalize_by_length, l1freqs, l2freqs, fv, limit=20):
        FeatureSet.__init__(self)
        self.dict_stot = dict_stot
        self.dict_ttos = dict_ttos
        self.normalize_by_length = normalize_by_length
        self.l1freqs = l1freqs
        self.l2freqs = l2freqs
        self.fv = fv
        self.limit = limit

    def run(self, segment_pair):
        features = []
        features.extend(FeatureQmax.feature_dict_qmax_nosmooth_zipf_freq(segment_pair.source_tok, segment_pair.target_tok, self.dict_stot, self.normalize_by_length, self.l2freqs, self.fv, self.limit))
        features.extend(FeatureQmax.feature_dict_qmax_nosmooth_zipf_freq(segment_pair.target_tok, segment_pair.source_tok, self.dict_ttos, self.normalize_by_length, self.l1freqs, self.fv, self.limit))
        return features

    @staticmethod
    def feature_dict_qmax_nosmooth_nolimit(slwords, tlwords, dict_stot, normalize_by_length, fv):
        logresult = 0
    
        slwords.append("NULL")
        slwords_s = [s for s in set(slwords) if s in dict_stot.d]
        tlwords2 = list(tlwords)
    
        count_t_in_dict = 0
        for tlword in tlwords2:
            if tlword in dict_stot.dinv:
                t = [dict_stot.get_prob_alpha(slword, tlword) for slword in slwords_s]
                #t.extend([dict_stot.get_prob_nonalpha(slword, tlword) for slword in slwords_s_n])
                prob = max(t, default=dict_stot.smooth)
                logresult += math.log(prob)
                logging.debug("\t"+str(prob)+"\t"+str(logresult))
                count_t_in_dict += 1
    
        if normalize_by_length:
            if fv >= 2:
                if count_t_in_dict > 0:
                    logresult = float(logresult) / float(
                        count_t_in_dict)  # the max is to prevent zero division when tl sentence is empty
                else:
                    return -1 # no word in T could be found in the dictionary, so this feature cannot be computed and the funciton returns -1
            else:
                # old behavior (it was a bug)
                logresult = float(logresult) / float(count_t_in_dict)
        return math.exp(logresult)

    @staticmethod
    def feature_dict_qmax_nosmooth_zipf_freq(slwords, tlwords, dict_stot, normalize_by_length, freqs, fv, limit):
        t_word_splits = freqs.split_sentence_by_freq(tlwords[0:limit])
        output = []
        for i in range(0, 4):
            output.append(FeatureQmax.feature_dict_qmax_nosmooth_nolimit(slwords, t_word_splits[i], dict_stot, normalize_by_length, fv))
        return output


class FeatureQmaxOld(FeatureSet):
    captions = ["qmax1", "qmax2"]

    def __init__(self, dict_stot, dict_ttos, normalize_by_length, treat_oovs, fv, limit=20):
        FeatureSet.__init__(self)
        self.dict_stot = dict_stot
        self.dict_ttos = dict_ttos
        self.normalize_by_length = normalize_by_length
        self.treat_oovs = treat_oovs
        self.dict_ttos = dict_ttos
        self.fv = fv
        self.limit = limit

    def run(self, segment_pair):
        features = []
        features.append(FeatureQmaxOld.feature_dict_qmax(segment_pair.source_tok, segment_pair.target_tok, self.dict_stot, self.normalize_by_length, self.treat_oovs, self.dict_ttos, self.fv, self.limit))
        features.append(FeatureQmaxOld.feature_dict_qmax(segment_pair.target_tok, segment_pair.source_tok, self.dict_ttos, self.normalize_by_length, self.treat_oovs, self.dict_ttos, self.fv, self.limit))
        return features

    # Qmax: probability product of correspondences found (using max probabilty
    # rather than alignment).
    @staticmethod
    def feature_dict_qmax(slwords, tlwords, dict_stot, normalize_by_length, treat_oovs, dict_ttos, fv, limit):
        logresult = 0
    
        slwords_s_a = set()
        slwords_s_n = set()
        for i in slwords:
            if regex_alpha.match(i):
                if i in dict_stot.d:
                    slwords_s_a.add(i)
            else:
                slwords_s_n.add(i)
    
        slwords_s_n.add("NULL")
        tlwords2 = list(tlwords)
        tlwords2.sort(key=len, reverse=True)
    
        if treat_oovs:
            for tlword in tlwords2[0:limit]:
                if tlword not in dict_ttos:
                    if fv >= 2:
                        logresult += math.log(0.0000001)
                    else:
                        pass  # old behavior (it was a bug)
                else:
                    t = [dict_stot.get_prob_alpha(slword, tlword) for slword in slwords_s_a]
                    t.extend([dict_stot.get_prob_nonalpha(slword, tlword) for slword in slwords_s_n])
                    prob = max(t, default=dict_stot.smooth)
                    logresult += math.log(prob)
        else:
            for tlword in tlwords2[0:limit]:
                t = [dict_stot.get_prob_alpha(slword, tlword) for slword in slwords_s_a]
                t.extend([dict_stot.get_prob_nonalpha(slword, tlword) for slword in slwords_s_n])
                prob = max(t, default=dict_stot.smooth)
                logresult += math.log(prob)
    
        if normalize_by_length:
            if fv >= 2:
                logresult = float(logresult) / float(max(1, min(len(tlwords),
                                                                limit)))  # the max is to prevent zero division when tl sentence is empty
            else:
                # old behavior (it was a bug)
                logresult = float(logresult) / float(max(len(tlwords), limit))
    
        return math.exp(logresult)


class FeatureLengthPoisson(FeatureSet):
    captions = ["poisson1", "poisson2"]

    def __init__(self, length_ratio):
        FeatureSet.__init__(self)
        self.length_ratio = length_ratio

    def run(self, segment_pair):
        features = []
        features.append(FeatureLengthPoisson.feature_length_poisson(segment_pair.source_tok, segment_pair.target_tok, self.length_ratio))
        features.append(FeatureLengthPoisson.feature_length_poisson(segment_pair.target_tok, segment_pair.source_tok, 1.0/self.length_ratio))
        return features

    # Likelihood of a particular sentence length ratio given the overall mean
    # and the Poisson probability distribution.
    @staticmethod
    def feature_length_poisson(slsentence, tlsentence, ratio):
        #try:
        sllen = max(len(slsentence), 0.1)
        tllen = max(len(tlsentence), 0.1)
        return math.exp(-sllen*ratio + tllen*math.log(sllen*ratio) - FeatureLengthPoisson.log_stirling(tllen))
        #except:
        #    logging.warning("Overflow when computing feature_length_poisson")
        #    return FeatureLengthPoisson.feature_length_poisson(["a"],["a"],1.0)

    @staticmethod
    def log_stirling(n):
        fn = float(n)
        fn = n * math.log(fn) + math.log(2 * math.pi * fn) / 2 - fn + \
                     (fn ** -1) / 12 - (fn ** -3) / 360 + (fn ** -5) / 1260 - \
                     (fn ** -7) / 1680 + (fn ** -9) / 1188
        return(fn)

class FeatureCharacterMeasurements(FeatureSet):
    captions = ["ndif1", "freq11", "freq21", "freq31",
                "ent1", "maxrep1", "maxword1",
                "ndif2", "freq12", "freq22", "freq32",
                "ent2", "maxrep2", "maxword2"]

    def __init__(self):
        FeatureSet.__init__(self)

    def run(self, segment_pair):
        features = []
        features.extend(FeatureCharacterMeasurements.feature_character_measurements(segment_pair.source))
        features.extend(FeatureCharacterMeasurements.feature_character_measurements(segment_pair.target))
        return features

    @staticmethod
    def feature_character_measurements(sentence):
        res = []
    
        # Number of different characters
        res.append(FeatureSet.logistic(len(set(sentence)), 40, 0.4))
    
        # Number of 3 most frequent characters, normalized by sentence length
        c = Counter(list(sentence)).most_common(3)
    
        while len(c) != 3:
            c.append(("", 0))
    
        res.extend([float(i[1])/(1.0 + float(len(sentence))) for i in c])
    
        # Entropy of the string
        res.append(FeatureSet.logistic(FeatureCharacterMeasurements.entropy(sentence), 5, 0.2))
    
        # Max number of consecutive repetitions of the same character
        repetitions = [len(i[0]) for i in re_repetition.findall(sentence)]
        x = 0 if len(repetitions) == 0 else max(repetitions)
        res.append(FeatureSet.logistic(x, 5, 0.7))
    
        # Length of the longest pseudo-word (blanks only)
        lengths = [len(i) for i in sentence.split(" ")]
        x = 0 if len(lengths) == 0 else max(lengths)
        res.append(FeatureSet.logistic(x, 30, 0.4))
    
        return res

    @staticmethod
    def entropy(s):
        l = float(len(s))
        return -sum(map(lambda a: (a/l)*math.log2(a/l), Counter(s).values()))



class FeatureCharacterClassDist(FeatureSet):
    captions = ["npunct1", "narabic1", "ngreek1", "nhan1",
                "nhangul1", "nhebrew1", "nlatin1", "nlatin_e_a1",
                "nlatin_e_b1", "nlatin_sup1", "nbasic_latin1",
                "nhiragana1", "nkatakana1", "ncyrillic1",
                "ndevanagari1", "nmalayalam1", "nother1",
                "npunct2", "narabic2", "ngreek2", "nhan2",
                "nhangul2", "nhebrew2", "nlatin2", "nlatin_e_a2",
                "nlatin_e_b2", "nlatin_sup2", "nbasic_latin2",
                "nhiragana2", "nkatakana2", "ncyrillic2",
                "ndevanagari2", "nmalayalam2", "nother2"]

    def __init__(self):
        FeatureSet.__init__(self)

    def run(self, segment_pair):
        features = []
        features.extend(FeatureCharacterClassDist.feature_character_class_dist(segment_pair.source))
        features.extend(FeatureCharacterClassDist.feature_character_class_dist(segment_pair.target))
        return features

    # Measure number of characters of main unicode classes
    @staticmethod
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
    
        return [FeatureSet.logistic(x, 100, 0.4) for x in result]


class FeatureSentenceLength(FeatureSet):
    captions = ["length1","length2"]

    def __init__(self):
        FeatureSet.__init__(self)

    def run(self, segment_pair):
        features = []
        features.append(FeatureSentenceLength.feature_sentence_length(segment_pair.source))
        features.append(FeatureSentenceLength.feature_sentence_length(segment_pair.target))
        return features

    # Sentence length, normalized using a logistic function
    @staticmethod
    def feature_sentence_length(sentence):
        x = float(len(sentence))
        return FeatureSet.logistic(x, 40, 0.1)

class FeatureSentenceLengthTok(FeatureSentenceLength):
    captions = ["ntok1","ntok2"]

    def __init__(self):
        FeatureSentenceLength.__init__(self)

    def run(self, segment_pair):
        features = []
        features.append(FeatureSentenceLength.feature_sentence_length(segment_pair.source_tok))
        features.append(FeatureSentenceLength.feature_sentence_length(segment_pair.target_tok))
        return features

class FeatureLangDetection(FeatureSet):
    captions = ["lang1","lang2"]

    def __init__(self, code1, code2):
        FeatureSet.__init__(self)
        self.code1 = code1
        self.code2 = code2

    def run(self, segment_pair):
        features = []
        features.append(FeatureLangDetection.feature_language(segment_pair.source, self.code1))
        features.append(FeatureLangDetection.feature_language(segment_pair.target, self.code2))
        return features

    # Language detected
    @staticmethod
    def feature_language(sentence, code):
        if code=="nb":
            code = "no"
    
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
        else:
            score = float(details[0][2])/100.0
    
        if details[0][1] != code:
            if code=="gl" and  (details[0][1] == "pt" or details[0][1] == "es"):
                return score
            if code=="no" and details[0][1] == "da":
                return score
            if code=="nn" and (details[0][1] == "no" or details[0][1] == "da"):
                return score
            else:
                return 0.0
        else:
            return score


class Features(object):

    def __init__(self, stokenizer, ttokenizer):
        self.featclasses = []
        self.stokenizer = stokenizer
        self.ttokenizer = ttokenizer

    def add_featclass(self, fclass):
        self.featclasses.append(fclass)

    def run(self, source_seg, target_seg):
        segment_pair = SegmentPair(source_seg, target_seg, self.stokenizer, self.ttokenizer)
        feats = []
        for feat_gen in self.featclasses:
            #logging.info(feat_gen.captions)
            features = feat_gen.run(segment_pair)
            feats.extend(features)
        return feats

    def get_feat_captions(self):
        captions = []
        for feat_gen in self.featclasses:
            captions.extend(feat_gen.captions)
        return captions

#    def __str__(self):
#        
#        print("TITLES: {}".format(len(self.titles)))
#        print("FEATURES: {}".format(len(self.feat)))
#        result = []
#        for i in range(len(self.titles)):
#            result.append("{}: {}".format(self.titles[i], self.feat[i]))
#        
#        return "\n".join(result)  
        
def build_features_generator(tokenize_l, tokenize_r, args):
    features = Features(tokenize_l, tokenize_r)

    length_ratio = args.length_ratio
    dict12 = args.dict_sl_tl
    dict21 = args.dict_tl_sl
    if args.sl_word_freqs != None:
        l1freqs = args.sl_word_freqs
    if args.tl_word_freqs != None:
        l2freqs = args.tl_word_freqs
    normalize_by_length = args.normalize_by_length
    qmax_limit = args.qmax_limit
    treat_oovs = args.treat_oovs
    disable_features_quest = args.disable_features_quest
    disable_features_lang = args.disable_lang_ident
    lang1 = args.source_lang
    lang2 = args.target_lang
    fv    = args.features_version

    #if not disable_features_lang:
    #    features.add_featclass(FeatureLangDetection(lang1,lang2))
    features.add_featclass(FeatureSentenceLength())
    #features.add_featclass(FeatureCharacterClassDist())
    #features.add_featclass(FeatureCharacterMeasurements())
    features.add_featclass(FeatureSentenceLengthTok())
    features.add_featclass(FeatureLengthPoisson(length_ratio))

    if fv < 4:
        features.add_featclass(FeatureQmaxOld(dict12, dict21, normalize_by_length, treat_oovs, fv))
        features.add_featclass(FeautreDicCoverage(dict12, dict21))
    else:
        # Feature version 4 using cummulated probabilities in qmax and word frecuencies
        features.add_featclass(FeatureQmax(dict12, dict21, normalize_by_length, l1freqs, l2freqs, fv))

        features.add_featclass(FeautreDicCoverage(dict12, dict21))
        features.add_featclass(FeautreDicCoverageQuartiles(dict12, dict21, l1freqs, l2freqs))

    if disable_features_quest:
        # Average token length
        #features.add_featclass(FeatureAvgWordLneght())

        # Number of punctuation marks
        #features.add_featclass(FeatureNumPunctMarks())

        # Number of punctuation marks of each type: dot, comma, colon,
        # semicolon, double quotes, single quotes
        features.add_featclass(FeatureNumPunctMarksType())

        # Numeric expression preservation
        features.add_featclass(FeatureNumberPreservation())

        # Capitalized letter preservation
        features.add_featclass(FeatureCapitalizedPreservation())

    #Add LM features
    if args.lm_file_sl and args.lm_file_tl:
        features.add_featclass(FeatureLM(lang1, lang2, tokenize_l, tokenize_r, args.lm_file_sl, args.lm_file_tl))

    return features
