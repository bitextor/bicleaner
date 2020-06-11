#!/usr/bin/env python
import gzip
import math


# Class to store word freqences. Word frequences are read from a tab-sepparated file containing two fields: freqences
# first and words second. Words must be lowercased. The file must be gzipped. Such files can be easyly produced from
# monolingual text running a command like this:
# cat monolingual.txt | tokenizer.sh | tr ' ' '\n' | tr '[:upper:]' '[:lower:]' | sort | uniq -c > wordfreq.txt
class WordZipfFreqDist(object):

    # Constructor
    def __init__(self, file_with_freq):
        self.word_freqs = dict()
        fname = file_with_freq if not hasattr(file_with_freq, 'name') else file_with_freq.name
        word_ocss = dict()
        with gzip.open(fname, "r") as reader:
            for line in reader:
                line = line.decode().strip()
                parts = line.split()
                word = parts[-1]
                occs = int(parts[0])
                word_ocss[word] = occs
        self.total_words = sum(word_ocss.values())
        for word, occs in word_ocss.items():
            self.word_freqs[word] = int(math.log(float(occs)/float(self.total_words))*100)
        self.min_freq = int(math.log(1.0/float(self.total_words))*100)
        max_val = max(self.word_freqs.values())
        min_max_diff = abs(max_val)-abs(self.min_freq)
        self.q1limit = self.min_freq-min_max_diff
        self.q2limit = self.min_freq-(2*min_max_diff)
        self.q3limit = self.min_freq-(3*min_max_diff)

    def split_sentence_by_freq(self, sentence):
        word_splits = dict()
        for i in range(0, 4):
            word_splits[i] = set()

        for w in sentence:
            word_splits[self.get_word_quartile(w)-1].add(w)

        return word_splits

    def get_word_quartile(self, word):
        if word in self.word_freqs:
            val_word = self.word_freqs[word]
            if val_word <= self.q1limit:
                return 1
            elif val_word <= self.q2limit:
                return 2
            elif val_word <= self.q3limit:
                return 3
            else:
                return 4
        else:
            return 4

    def word_is_in_q1(self, word):
        if word in self.word_freqs:
            val_word = self.word_freqs[word]
            return val_word <= self.q1limit
        else:
            return False

    def word_is_in_q2(self, word):
        if word in self.word_freqs:
            val_word = self.word_freqs[word]
            return val_word <= self.q2limit
        else:
            return False

    def word_is_in_q3(self, word):
        if word in self.word_freqs:
            val_word = self.word_freqs[word]
            return val_word <= self.q3limit
        else:
            return False

    def word_is_in_q4(self, word):
        if word in self.word_freqs:
            val_word = self.word_freqs[word]
            return val_word > self.q3limit
        else:
            return True

    def get_word_freq(self, word):
        word = word.lower()
        if word in self.word_freqs:
            return self.word_freqs[word]
        else:
            return self.min_freq
