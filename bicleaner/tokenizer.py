from sacremoses import MosesTokenizer
from toolwrapper import ToolWrapper
from subprocess import run, PIPE
import logging
import sys
import os

try:
    from .util import no_escaping
except (SystemError, ImportError):
    from util import  no_escaping


class Tokenizer:
    def __init__(self, command=None,  l="en"):
        if command:
            self.cmd = command.split(' ')
            self.tokenizer = ToolWrapper(self.cmd)
            self.external =  True
            self.spm = command.find('spm_encode') > -1
        else:
            self.tokenizer = MosesTokenizer(lang=l)
            self.external = False
            self.spm = False
            self.cmd = None

    def tokenize(self, text):
        if self.external:
            if isinstance(text, list):
                return self.tokenize_block('\n'.join(text) + '\n').split('\n')
            else:
                self.tokenizer.writeline(text.rstrip('\n'))
                return ([no_escaping(t) for t in self.tokenizer.readline().rstrip('\n').split()])
        else:
            if isinstance(text, list):
                return [self.tokenizer.tokenize(line, escape=False) for line in text]
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

    def start(self):
        if self.external:
            self.tokenizer.start()

    def restart(self):
        if self.external:
            self.tokenizer.restart()

    def tokenize_block(self, text):
        logging.debug(f'Opening subprocess: {self.cmd!r}')
        output = run(self.cmd, input=text, stdout=PIPE, stderr=PIPE, env=os.environ, encoding='utf-8')
        if output.returncode != 0:
            logging.error(output.stderr)
            sys.exit(1)
        else:
            logging.debug(f'Succesfully subprocess: {self.cmd!r}')
            logging.debug(f'Errors: {output.stderr}')
            return output.stdout
