#!/usr/bin/env python


__author__ = "Marta Bañón"
__version__ = "Version 0.1 # 28/09/2018 # Classifier test # Marta Bañón"

import subprocess
import bicleaner

def setup_function():
	langpackurl = "https://github.com/bitextor/bitextor-data/raw/master/bicleaner/en-de.tar.gz"
	tar = "tar -xzf en-de.tar.gz"
	command = "cd bicleaner && mkdir -p lang && cd lang && wget -q {0} && {1}  && cd ../..".format(langpackurl, tar)	
	p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
	p.wait()

def teardown_function():
	command = "rm -r bicleaner/lang && rm tests/test-corpus.en-de.classified"
	p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
	p.wait()

def bicleaner_test():
	bicleaner_cmd = "python3 bicleaner/bicleaner-classifier-full.py \
      tests/test-corpus.en-de  \
      tests/test-corpus.en-de.classified  \
      -m bicleaner/lang/en-de/training.en-de.yaml \
      -b 100  -q"

	p = subprocess.Popen(bicleaner_cmd, shell=True, stdout=subprocess.PIPE)
	p.wait()

	scores = []
	
	with open("tests/test-corpus.en-de.classified", "r") as classified_file:
		for line in classified_file:	
			line = line.rstrip("\n")
			print(line)

			try:
				url1, url2, source_sentence, target_sentence, score = line.split('\t')
				scores.append(round(float(score), 1))					
			except Exception as e:
				print(e)
				scores.append("-1")
				continue
	return scores
	
def test_results():
	expected = [0, 0, 0, 0.8, 0, 0.6, 0, 0, 0.4, 0]
	results = bicleaner_test()

	for  i in range(len(expected)):
		assert(results[i] == expected[i])
