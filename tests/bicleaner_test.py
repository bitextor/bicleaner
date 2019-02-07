#!/usr/bin/env python


__author__ = "Marta Bañón"
__version__ = "Version 0.1 # 28/09/2018 # Classifier test # Marta Bañón"

import subprocess
import bicleaner

def setup_function():
	print("Running test setup...")
	langpackurl = "https://github.com/bitextor/bitextor-data/releases/download/bicleaner-v1.0/en-de.tar.gz"
	tar = "tar -xzf en-de.tar.gz"
	command = "mkdir -p test_langpacks && cd test_langpacks && wget -q {0} && {1}  && cd ../..".format(langpackurl, tar)	
	p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
	p.wait()

def teardown_function():
	print("Running test teardown...")
	command = "rm -r test_langpacks && rm tests/test-corpus.en-de.classified"
	p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
	p.wait()

def bicleaner_test():
	print("Running test body...")
	bicleaner_cmd = "bicleaner-classify  \
      tests/test-corpus.en-de  \
      tests/test-corpus.en-de.classified  \
      test_langpacks/en-de/training.en-de.yaml -q"

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

	expected = [0, 0, 0, 0, 0, 0.6, 0, 0, 0.4, 0]
	results = bicleaner_test()
	print("Checking test results...")
	for  i in range(len(expected)):
		assert(results[i] == expected[i])
	
