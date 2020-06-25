
from toolwrapper import ToolWrapper
from sacremoses import MosesTokenizer

try:
    from .util import no_escaping
except (SystemError, ImportError):
    from util import  no_escaping


class Tokenizer:
    def __init__(self, command=None,  l="en"):
        if command:        
            self.tokenizer=ToolWrapper(command.split(' '))
            self.external =  True         
        else:
            self.tokenizer = MosesTokenizer(lang=l)
            self.external = False
          
    
    def tokenize(self, text):
        if self.external:
            self.tokenizer.writeline(text.rstrip('\n'))
            return ([no_escaping(t) for t in self.tokenizer.readline().rstrip('\n').split()])
        else:   
            return self.tokenizer.tokenize(text, escape=False)

    def close(self):
        if self.external:
            try:
                self.tokenizer.close()                
            except:
                return