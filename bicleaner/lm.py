import kenlm
from enum import Enum
from util import MosesTokenizer
from mosestokenizer import MosesPunctuationNormalizer, MosesSentenceSplitter
from tempfile import TemporaryFile, NamedTemporaryFile
import subprocess
import shutil
import os
import argparse
import logging
import numpy
import regex

class LMType(Enum):
    #Needed for argparse
    PLACEHOLDER='PLACEHOLDER'
    CHARACTER='CHARACTER'
    
    def __str__(self):
        return self.value


class UnicodeWordClassifier:
    regex_basic_latin      = regex.compile("^[\p{InBasic_Latin}]+$")
    regex_latin_supplement = regex.compile("^[\p{InLatin-1_Supplement}\p{InBasic_Latin}]+$")
    regex_latin_extended = regex.compile("^[\p{InLatin-1_Supplement}\p{InBasic_Latin}\p{InLatin_Extended-A}\p{InLatin_Extended-B}]+$")
    regex_arabic           = regex.compile("^[\p{Arabic}]+$")
    regex_greek            = regex.compile("^[\p{Greek}]+$")
    regex_cyrillic         = regex.compile("^[\p{Cyrillic}]+$")
    regexes =[ ('BASIC_LATIN',regex_basic_latin) , ('LATIN_SUPPLEMENT',regex_latin_supplement) ,  ('LATIN_EXTENDED',regex_latin_extended),
              ('ARABIC',regex_arabic), ('GREEK',regex_greek), ('CYRILIC',regex_cyrillic)]
    
    @classmethod
    def classify_word(cls,word):
        for name,r in cls.regexes:
            if r.match(word):
                return name
        return "OTHER"
            

class LMFluencyFilter:
    
    def __init__(self, lm_type:LMType , language:str):
        """
            lm_type: LMType
            language: language code
        """
        
        self.language=language
        self.tokenizer=MosesTokenizer(self.language)
        self.normalizer=MosesPunctuationNormalizer(self.language)
        self.splitter=MosesSentenceSplitter(self.language, more=False)
        self.type=lm_type
    
    @classmethod
    def _ispunctuation(cls,t):
        return all( not c.isalnum() for c in t)
    
    @classmethod
    def _replace_placeholder(cls,t):
        if t.isalpha():
            unicodeGroup = UnicodeWordClassifier.classify_word(t)
            if t.islower():
                return "TOKEN:ALPHA:LOWER:"+unicodeGroup
            elif t.istitle():
                return "TOKEN:ALPHA:TITLE:"+unicodeGroup
            elif t.isupper():
                return "TOKEN:ALPHA:UPPER:"+unicodeGroup
            else:
                return "TOKEN:ALPHA:MIXED:"+unicodeGroup
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
    
    def _sentence_split(self,sentence:str):
        return self.splitter([sentence])
    
    def _tokenize(self, sentence):
        sentence=self.normalizer(sentence)
        if self.type != LMType.CHARACTER:
            tokline=" ".join(self.tokenizer(sentence))
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
                sentences=self._sentence_split(line)
                for s in sentences:
                    tokline=self._tokenize(s)
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
            params="-o 7 --discount_fallback"
    
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
   
    @classmethod 
    def estimate_threshold(cls,filter_a,filter_b, dev_corpus_a:str,  dev_corpus_b:str):
        scores=[]
        with open(dev_corpus_a) as corpus_a_f, open(dev_corpus_b) as corpus_b_f:
            for linea,lineb in zip(corpus_a_f,corpus_b_f):
                linea=linea.rstrip("\n")
                lineb=lineb.rstrip("\n")
                scores.append(filter_a.score(linea)+filter_b.score(lineb))
        return numpy.mean(scores),numpy.std(scores)
        
    
    def score(self, sentence:str):
        #We need to preprocess the sentence in the same way as when training the LM
        sents= self._sentence_split(sentence)
        processed_sents=[self._introduce_placeholders(self._tokenize(s)) for s in sents]
        logging.debug("Scoring: {}".format(processed_sents))
        #TODO: we will estimate threshold later
        raw_scores= [self._raw_score(s) for s in processed_sents]
        
        #Normalize score
        return sum(raw_scores)/(sum([len(s.split()) for s in processed_sents]) + len(processed_sents) ) # We divide by total number of tokens + 1 for each sentence (taken from kenlm perplexity method)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language",required=True)
    parser.add_argument("--language_b")
    parser.add_argument("--lm_type",type=lambda t: LMType[t], choices=list(LMType),required=True)
    parser.add_argument("--train",action='store_true')
    parser.add_argument("--score",action='store_true')
    parser.add_argument("--stats",action='store_true')
    parser.add_argument("--corpus")
    parser.add_argument("--corpus_b")
    parser.add_argument("--lm_file")
    parser.add_argument("--lm_file_b")
    
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
    if args.stats:
        ff.load_lm(args.lm_file)
        ff_b=LMFluencyFilter(args.lm_type, args.language_b)
        ff_b.load_lm(args.lm_file_b)
        mean,stdev=LMFluencyFilter.estimate_threshold(ff,ff_b,args.corpus,args.corpus_b)
        print("{} {}".format(mean,stdev))

