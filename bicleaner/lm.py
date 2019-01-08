import kenlm
from enum import Enum
from enum import auto
from mosestokenizer import MosesTokenizer
from tempfile import TemporaryFile, NamedTemporaryFile
import subprocess
import shutil
import os
import argparse
import logging

class LMType(Enum):
    #Needed for argparse
    PLACEHOLDER='PLACEHOLDER'
    CHARACTER='CHARACTER'
    
    def __str__(self):
        return self.value

class LMFluencyFilter:
    
    def __init__(self, lm_type:LMType , language:str):
        """
            lm_type: LMType
            language: language code
        """
        
        self.language=language
        self.tokenizer=MosesTokenizer(self.language)
        self.type=lm_type
    
    @classmethod
    def _ispunctuation(cls,t):
        return all( not c.isalnum() for c in t)
    
    @classmethod
    def _replace_placeholder(cls,t):
        if t.isalpha():
            if t.islower():
                return "TOKEN:ALPHA:LOWER"
            elif t.istitle():
                return "TOKEN:ALPHA:TITLE"
            elif t.isupper():
                return "TOKEN:ALPHA:UPPER"
            else:
                return "TOKEN:ALPHA:MIXED"
        else:
            if t.isnumeric():
                return "TOKEN:NUMERIC"
            elif cls._ispunctuation(t):
                return t
            else:
                return "TOKEN:MIXED"
    
    @classmethod
    def _estimate_kenlm(cls, corpus:str, lm_file:str, params:str):
        subprocess.run("lmplz "+params+" < "+corpus+" > "+lm_file+".arpa", shell=True)
        subprocess.run("build_binary "+lm_file+".arpa "+ lm_file, shell=True)
    
    def load_lm(self, lm_path:str):
        self.lm_path=lm_path
        self.lm=kenlm.LanguageModel(self.lm_path)
    
    def _tokenize(self, sentence):
        if self.type != LMType.CHARACTER:
            self.tokenizer.writeline(sentence)
            tokline=self.tokenizer.readline().rstrip("\n")
        else:
            tokline=" ".join([ "SPACE" if c == " " else c for c in sentence  ])
        return tokline
    
    def _introduce_placeholders(self, sentence):
        if self.type != LMType.PLACEHOLDER:
            return sentence
        else:
            toks=[ self._replace_placeholder(t) for t in sentence.split() ]
            return " ".join(toks)
    
    def train_lm(self, text_path:str):
        tokenized_f=NamedTemporaryFile("w", delete=False)
        placeholderized_f=NamedTemporaryFile("w", delete=False)
        
        #Tokenize text
        with open(text_path) as input_f:
            for line in input_f:
                line=line.rstrip("\n")
                tokline=self._tokenize(line)
                tokenized_f.write(tokline)
                tokenized_f.write("\n")
        tokenized_f.close()
            
        #Perform placeholder replacement if needed
        with open(tokenized_f.name) as tokenized_ff:
            for line in tokenized_ff:
                line=line.rstrip("\n")
                with_placeholders=self._introduce_placeholders(line)
                logging.debug("Processed training example: {}".format(with_placeholders))
                placeholderized_f.write(with_placeholders)
                placeholderized_f.write("\n")
        placeholderized_f.close()
        
        #Estimate LM
        lm_file=NamedTemporaryFile(delete=False)
        lm_file.close()
        
        if self.type == LMType.CHARACTER:
            params="-o 7 --discount_fallback"
        else:
            params="-o 7"
    
        self._estimate_kenlm(placeholderized_f.name, lm_file.name,params)
        self.lm_path=lm_file.name
        
        self.lm=kenlm.LanguageModel(self.lm_path)
        
        #Remove temporary files
        os.remove(tokenized_f.name) 
        os.remove(placeholderized_f.name)
    
    def copy_lm(self,dst:str):
        shutil.copyfile(self.lm_path, dst)
    
    def cleanup(self):
        os.remove(self.lm_path)
        
    def _raw_score(self, sentence:str):
        return self.lm.score(sentence)
    
    def estimate_threshold(self):
        pass
    
    def score(self, sentence:str):
        #We need to preprocess the sentence in the same way as when training the LM
        processed_sentence=self._introduce_placeholders(self._tokenize(sentence))
        logging.debug("Scoring: {}".format(processed_sentence))
        #TODO: we will estimate threshold later
        return self._raw_score(processed_sentence)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language",required=True)
    parser.add_argument("--lm_type",type=lambda t: LMType[t], choices=list(LMType),required=True)
    parser.add_argument("--train",action='store_true')
    parser.add_argument("--score",action='store_true')
    parser.add_argument("--corpus")
    parser.add_argument("--lm_file")
    
    parser.add_argument("--debug",action='store_true')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    ff = LMFluencyFilter(args.lm_type, args.language)
    
    if args.train:
        ff.train_lm(args.corpus)
        ff.copy_lm(args.lm_file)
        ff.cleanup()
    
    if args.score:
        ff.load_lm(args.lm_file)
        with open(args.corpus) as corpus_f:
            for line in corpus_f:
                line=line.rstrip("\n")
                print(ff.score(line))
