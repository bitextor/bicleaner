#!/usr/bin/env python
try:
    from .word_freqs_zipf import WordZipfFreqDist
except (ImportError, SystemError):
    from word_freqs_zipf import WordZipfFreqDist

# Class to store word frequences. Word frequences are read from a tab-sepparated file containing two fields: freqences
# first and words second. Words must be lowercased. The file must be gzipped. Such files can be easyly produced from
# monolingual text running a command like this:
# cat monolingual.txt | tokenizer.sh | tr ' ' '\n' | tr '[:upper:]' '[:lower:]' | sort | uniq -c > wordfreq.txt
class WordZipfFreqDistDoubleLinked(WordZipfFreqDist):

    # Constructor
    def __init__(self, file_with_freq):
        WordZipfFreqDist.__init__(self,file_with_freq)
        self.freq_words = dict()
        for w, f in self.word_freqs.items():
            if f not in self.freq_words:
                self.freq_words[f] = set()
            self.freq_words[f].add(w)

    def get_words_for_freq(self, freq):
        if freq in self.freq_words:
            return self.freq_words[freq]
        else:
            return None
