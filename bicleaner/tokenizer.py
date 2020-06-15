#!/usr/bin/env python

from toolwrapper import ToolWrapper
from sacremoses import MosesTokenizer

try:
    from .util import no_escaping
except (SystemError, ImportError):
    from util import  no_escaping


class Tokenizer:
    def __init__(self, path=None,  l="en"):
        if args.path:        
            tokenizer=ToolWrapper(args.path.split(' '))
            external =  True         
        else:
            tokenizer = MosesTokenizer(lang=l)
            external = False
          
    
    def tokenize(self, text):
        if self.external:
            self.tokenizer.writeline(text.rstrip('\n'))
            return ([no_escaping(t) for t in self.tokenizer.readline().rstrip('\n').split())
        else:   
            return self.tokenizer.tokenize(text, escape=False)
                