#!/usr/bin/env python


__author__ = "Marta Bañón"
__version__ = "Version 0.1 # 28/09/2018 # Classifier test # Marta Bañón"

import subprocess
import bicleaner

def setup_module():

	print("Running test setup...")
	
	langpackurl = "https://github.com/bitextor/bicleaner-data/releases/latest/download/en-de.tar.gz"
	tar = "tar -xzvf en-de.tar.gz"
	command = "mkdir -p test_langpacks && cd test_langpacks && wget -q {0} && {1}  && cd ..".format(langpackurl, tar)	
	p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
	p.wait()

	p = subprocess.Popen("cat test_langpacks/en-de/en-de.yaml |grep -v '_lm' | grep -v 'lm_type'  | grep -v '_perp' >  test_langpacks/en-de/en-de.nolm.yaml", shell=True, stdout=subprocess.PIPE)
	p.wait()
	
	

def teardown_module():
	print("Running test teardown...")
		
	command = "rm -r test_langpacks && rm tests/test-corpus.en-de.classified"
	p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
	p.wait()
	
	
def bicleaner_test(executable, training_yaml ):
	print("Running test body...")
	bicleaner_cmd = "{}  \
      tests/test-corpus.en-de  \
      tests/test-corpus.en-de.classified  \
      test_langpacks/en-de/{} -q".format(executable,training_yaml)

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

def test_full_process():
	expected = [0, 0, 0, 0, 0, 0.5, 0, 0.1, 0.3, 0]
	results = bicleaner_test("bicleaner-classify","en-de.yaml")
	print("Checking test results...")
	for  i in range(len(expected)):
		assert(results[i] == expected[i])
	
def test_full_process_nolm():
	expected = [0, 0, 0, 0, 0, 0.5, 0, 0.1, 0.3, 0]
	results = bicleaner_test("bicleaner-classify","en-de.nolm.yaml")
	print("Checking test results...")
	for  i in range(len(expected)):
		assert(results[i] == expected[i])

def test_lite_process():
        expected = [0, 0, 0, 0, 0, 0.5, 0, 0.1, 0.3, 0]
        results = bicleaner_test("bicleaner-classify-lite","en-de.yaml")
        print("Checking test results...")
        for  i in range(len(expected)):
                assert(results[i] == expected[i])


def test_lite_process_nolm():
        expected = [0, 0, 0, 0, 0, 0.5, 0, 0.1, 0.3, 0]
        results = bicleaner_test("bicleaner-classify-lite","en-de.nolm.yaml")
        print("Checking test results...")
        for  i in range(len(expected)):
                assert(results[i] == expected[i])
