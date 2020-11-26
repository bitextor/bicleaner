
from toolwrapper import ToolWrapper
from sacremoses import MosesTokenizer

try:
    from .util import no_escaping
except (SystemError, ImportError):
    from util import  no_escaping


class Tokenizer:
    def __init__(self, command=None,  l="en", pretokenized=False):
        self.pretokenized = pretokenized
        if pretokenized:
            self.command = None
            self.tokenizer = None
            self.external = True
            self.spm = False
        elif command:
            self.command = command
            self.tokenizer=ToolWrapper(command.split(' '))
            self.external =  True
            self.spm = command.find('spm_encode') > -1
        else:
            self.tokenizer = MosesTokenizer(lang=l)
            self.external = False
            self.spm = False

    def tokenize(self, text):
        if self.pretokenized:
            return text.split()
        elif self.external:
            self.tokenizer.writeline(text.rstrip('\n'))
            tmp_output = [t for t in self.tokenizer.readline().rstrip('\n').split()]
            output = []
            for t in tmp_output:
                if len(t) > 0:
                    output.append(no_escaping(t))
                else:
                    output.append("")
            return output
        else:
            return self.tokenizer.tokenize(text, escape=False)

    def detokenize(self, text):
        if self.spm:
            return ''.join(text).replace('\u2581',' ')
        else:
            return ' '.join(text)

    def close(self):
        if not self.pretokenized and self.external:
            try:
                self.tokenizer.close()
            except:
                return
