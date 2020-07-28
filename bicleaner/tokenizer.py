
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
            self.spm = command.find('spm_encode') > -1
        else:
            self.tokenizer = MosesTokenizer(lang=l)
            self.external = False
            self.spm = False

    def tokenize(self, text):
        if self.external:
            self.tokenizer.writeline(text.rstrip('\n'))
            return ([no_escaping(t) for t in self.tokenizer.readline().rstrip('\n').split()])
        else:
            return self.tokenizer.tokenize(text, escape=False)

    def detokenize(self, text):
        if self.spm:
            return ''.join(text).replace('\u2581',' ')
        else:
            return ' '.join(text)

    def close(self):
        if self.external:
            try:
                self.tokenizer.close()
            except:
                return
