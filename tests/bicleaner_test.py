__author__ = "Marta Bañón"
__version__ = "Version 0.1 # 28/09/2018 # Classifier test # Marta Bañón"

import subprocess
#from subprocess import check_output


def bicleaner_test():
	langpackurl = "https://github.com/bitextor/bitextor-data/raw/master/bicleaner/en-de.tar.gz"
	tar = "tar -xzf en-de.tar.gz"
	bicleaner_cmd = "python3 bicleaner/bicleaner-classifier-full.py \
      tests/test-corpus.en-de  \
      tests/test-corpus.en-de.classified  \
      -m bicleaner/lang/en-de/training.en-de.yaml \
      -b 100  -q"
	awk = "awk -F'\t' '$5 > 0 {print ;}' tests/test-corpus.en-de.classified"     

	command = "cd bicleaner && mkdir -p lang && cd lang && wget -q {0} && {1}  && cd ../.. && {2}".format(langpackurl, tar, bicleaner_cmd)
	p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
	p.wait()

	passed = 0

	with open("tests/test-corpus.en-de.classified", "r") as classified_file:
		for line in classified_file:	
			try:
				url1, url2, source_sentence, target_sentence, score = line.split('\t')
				if float(score) > 0.0:
					passed=passed+1
			except Exception as e:
				continue
	return passed			
	
#	out, err = p.communicate()
#	return out[-1]
#	return check_output(command)
	

def test_results():
	assert bicleaner_test() == 3