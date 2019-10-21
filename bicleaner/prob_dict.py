#!/usr/bin/env python
import logging
import gzip
import regex

#Allows to load modules while inside or outside the package
try:
    from .util import regex_alpha
except (SystemError, ImportError):
    from util import regex_alpha


# Probabilistic dictionary class
class ProbabilisticDictionary(object):

    # Constructor
    def __init__(self, file):
        self.alpha = regex_alpha
        fname = file if not hasattr(file, 'name') else file.name
        with gzip.open(fname, "rt") as fd:
            self.d = dict()
            self.minprob = 1.0 # minimum probability
            logging.debug("Loading dictionary {0}".format(fname))
            self.load(fd)
            logging.debug("Dictionary {0} loaded".format(fname))
            self.smooth = self.minprob*0.1  # smooth property

    # Method to load a dictionary to the class (called by the constructor)
    def load(self,fd):
        for line in fd:
            line  = line.rstrip("\n")
            parts = line.split()
            l1    = parts[1]
            l2    = parts[0]
            prob  = float(parts[2])
            if prob <= 0.0000001:
                continue
            if not self.alpha.match(l1):
                continue
            if prob < self.minprob:
                self.minprob = prob
            if l1 not in self.d:
                self.d[l1] = dict()
            self.d[l1][l2] = prob

    def get_prob(self, x, y):
        if self.alpha.match(x):
            if x not in self.d:
                return self.smooth
            elif y not in self.d[x]:
                return self.smooth
            else:
                return self.d[x][y]
        elif x == y: # relax this comparison?
            return 1.0
        else:
            return self.smooth


    def get_prob_alpha(self, x, y):
        # x is assumed to exist for performance reason!!!
        if y not in self.d[x]:
            return self.smooth
        else:
            return self.d[x][y]

    def get_prob_nonalpha(self, x, y):
        if x == y and len(x) < 20: # limit to 20 characters non-alphabetic words recognised by the dictionary
            return 1.0
        else:
            return self.smooth

    
    def __getitem__(self, key):
        if self.alpha.match(key):
            if key in self.d:
                return [i for i in self.d[key]]
            else:
                return []
        else:
            return [key]

    def __contains__(self, value):
        if self.alpha.match(value):
            return value in self.d
        else:
            return True
