#!/usr/bin/env python
import gzip


# Class to store word frequences. Word frequences are read from a tab-sepparated file containing two fields: freqences
# first and words second. Words must be lowercased. The file must be gzipped. Such files can be easyly produced from
# monolingual text running a command like this:
# cat monolingual.txt | tokenizer.sh | tr ' ' '\n' | tr '[:upper:]' '[:lower:]' | sort | uniq -c > wordfreq.txt
class WordFreqList(object):

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
            self.word_freqs[word] = float(occs)/float(self.total_words)
        self.min_freq = 1.0/float(self.total_words)

    def get_word_freq(self, word):
        word = word.lower()
        if word in self.word_freqs:
            return self.word_freqs[word]
        else:
            return self.min_freq
