
# bicleaner

![License](https://img.shields.io/badge/License-GPLv3-blue.svg)


Bicleaner (`bicleaner-classify`) is a tool in Python that aims at detecting noisy sentence pairs in a parallel corpus. It
indicates the likelihood of a pair of sentences being mutual translations (with a value near to 1) or not (with a value near to 0). Sentence pairs considered very noisy are scored with 0.

Although a training tool (`bicleaner-train`) is provided, you may want to use the available ready-to-use language packages. 
Please, visit https://github.com/bitextor/bicleaner-data/releases/latest or use `./utils/download-pack.sh` to download the latest language packages.
Visit our [Wiki](https://github.com/bitextor/bicleaner/wiki/How-to-train-your-Bicleaner) for a detailed example on Bicleaner training.

## Citation 

If you find Bicleaner useful, please consider citing the following papers:

> V. M. Sánchez-Cartagena, M. Bañón, S. Ortiz-Rojas and G. Ramírez-Sánchez,\
> "[Prompsit's submission to WMT 2018 Parallel Corpus Filtering shared task](http://www.statmt.org/wmt18/pdf/WMT116.pdf)",\
>in *Proceedings of the Third Conference on Machine Translation, Volume 2: Shared Task Papers*.\
>Brussels, Belgium: Association for Computational Linguistics, October 2018

```latex
@InProceedings{prompsit:2018:WMT,
  author    = { V\'{i}ctor M. S\'{a}nchez-Cartagena and Marta Ba{\~n}\'{o}n and Sergio Ortiz-Rojas and Gema Ram\'{i}rez-S\'{a}nchez},
  title     = {Prompsit's submission to WMT 2018 Parallel Corpus Filtering shared task},
  booktitle = {Proceedings of the Third Conference on Machine Translation, Volume 2: Shared Task Papers},
  month     = {October},
  address   = {Brussels, Belgium},
  publisher = {Association for Computational Linguistics}
}
```


> Gema Ramírez-Sánchez, Jaume Zaragoza-Bernabeu, Marta Bañón and Sergio Ortiz Rojas \
> "[Bifixer and Bicleaner: two open-source tools to clean your parallel data.](https://eamt2020.inesc-id.pt/proceedings-eamt2020.pdf#page=311)",\
>in *Proceedings of the 22nd Annual Conference of the European Association for Machine Translation*.\
>Lisboa, Portugal: European Association for Machine Translation, November 2020

```latex
@InProceedings{prompsit:2020:EAMT,
  author    = {Gema Ram\'{i}rez-S\'{a}nchez and Jaume Zaragoza-Bernabeu and Marta Ba{\~n}\'{o}n and Sergio Ortiz-Rojas},
  title     = {Bifixer and Bicleaner: two open-source tools to clean your parallel data.},
  booktitle = {Proceedings of the 22nd Annual Conference of the European Association for Machine Translation},
  pages	    = {291--298},
  isbn      = {978-989-33-0589-8},
  year	    = {2020},
  month     = {November},
  address   = {Lisboa, Portugal},
  publisher = {European Association for Machine Translation}
}
```

## Installation & Requirements

Bicleaner is written in Python and can be installed using `pip`:

```bash
python3.7 -m pip install bicleaner
```

Bicleaner Hard-rules requires the [KenLM](https://github.com/kpu/kenlm) Python bindings with support for 7-gram language models. You can easily install it by running the following commands:

```bash
git clone https://github.com/kpu/kenlm
cd kenlm
python3.7 -m pip install . --install-option="--max_order 7"
mkdir -p build && cd build
cmake .. -DKENLM_MAX_ORDER=7 -DCMAKE_INSTALL_PREFIX:PATH=/your/prefix/path
make -j all install
```

The remaining extra modules required by Bicleaner will be automatically downloaded and installed/upgraded (if required) with the first command.

After installation, three binary files (`bicleaner-train`, `bicleaner-classify` and `bicleaner-classify-lite`) will be located in your `python/installation/prefix/bin` directory. This is usually `$HOME/.local/bin` or `/usr/local/bin/`.

## Cleaning

`bicleaner-classify` aims at detecting noisy sentence pairs in a parallel corpus. It
indicates the likelihood of a pair of sentences being mutual translations (with a value near to 1) or not (with a value near to 0). Sentence pairs considered very noisy are scored with 0.

By default, the input file (the parallel corpus to be classified) must contain at least four columns, being:

* col1: URL 1
* col2: URL 2
* col3: Source sentence
* col4: Target sentence

but the source and target sentences column index can be customized by using the `--scol` and `--tcol` flags.

The generated output file will contain the same lines and columns that the original input file had, adding an extra column containing the Bicleaner classifier score.

This tool can be run with

```bash
bicleaner-classify [-h]
                   [-S SOURCE_TOKENIZER_COMMAND]
                   [-T TARGET_TOKENIZER_COMMAND] 
                   [--header]
                   [--scol SCOL]
                   [--tcol TCOL] 
                   [--tmp_dir TMP_DIR]
                   [-b BLOCK_SIZE] 
                   [-p PROCESSES] 
                   [-d DISCARDED_TUS]
                   [--lm_threshold LM_THRESHOLD] 
                   [--score_only]
                   [--disable_hardrules]
                   [--disable_lm_filter]
                   [--disable_porn_removal]
                   [--disable_minimal_length]
                   [-q] 
                   [--debug] 
                   [--logfile LOGFILE] 
                   [-v]
                   input 
                   [output] 
                   metadata
```

### Parameters

* positional arguments:
  * `input`: Tab-separated files to be classified (default line format: `URL1 URL2 SOURCE_SENTENCE TARGET_SENTENCE [EXTRA_COLUMNS]`, tab-separated). When input is -, reads standard input.
  * `output`: Output of the classification (default: standard output). When output is -, writes standard output.
  * `metadata`: Training metadata (YAML file), generated by `bicleaner-train` or [downloaded](https://github.com/bitextor/bicleaner-data/releases/latest) as a part of a language pack. You just need to `untar` the language pack for the pair of languages of the file you want to clean. The tar file contains the YAML metadata file.
  There's a script that can download and unpack it for you, use:
  ```bash
  $ ./utils/download-pack.sh en cs ./models
  ```
  to download English-Czech language pack to the ./models directory and unpack it.
* optional arguments:
  * `-h, --help`: show this help message and exit
* Optional:
  * `-S SOURCE_TOKENIZER_COMMAND`: Source language tokenizer full command (including flags if needed). If not given, Sacremoses tokenizer is used (with `escape=False` option).
  * `-T TARGET_TOKENIZER_COMMAND`: Target language tokenizer full command (including flags if needed). If not given, Sacremoses tokenizer is used (with `escape=False` option).
  * `--header`: Treats the first sentence of the input file as the header row. If set, the output will contain a header as well
  * `--scol SCOL`: Source sentence column (starting in 1). If `--header` is set, the expected value will be the name of the field (default: 3 if `--header` is not set else src_text)
  * `--tcol TCOL`: Target sentence column (starting in 1). If `--header` is set, the expected value will be the name of the field (default: 4 if `--header` is not set else trg_text)
  * `--tmp_dir TMP_DIR`: Temporary directory where creating the temporary files of this program (default: default system temp dir, defined by the environment variable TMPDIR in Unix)
  * `-b BLOCK_SIZE, --block_size BLOCK_SIZE`: Sentence pairs per block (default: 10000)
  * `p PROCESSES, --processes PROCESSES`: Number of processes to use (default: all CPUs minus one)
  * `-d DISCARDED_TUS, --discarded_tus DISCARDED_TUS`: TSV file with discarded TUs. Discarded TUs by the classifier are written in this file in TSV file. (default: None)
  * `--lm_threshold LM_THRESHOLD`: Threshold for language model fluency scoring. All sentence pairs whose LM fluency score falls below the threshold are removed (classifier score set to 0), unless the option --keep_lm_result is set. (default: 0.5)
  * `--score_only`: Only output one column which is the bicleaner score (default: False)
  * `--disable_hardrules`: Disables the bicleaner_hardrules filtering (only bicleaner_classify is applied) (default: False)
  * `--disable_lm_filter`: Disables LM filtering.
  * `--disable_porn_removal`: Disables porn removal.
  * `--disable_minimal_length` : Don't apply minimal length rule (default: False).

* Logging:
  * `-q, --quiet`: Silent logging mode (default: False)
  * `--debug`: Debug logging mode (default: False)
  * `--logfile LOGFILE`: Store log to a file (default: <_io.TextIOWrapper name='<stderr>' mode='w' encoding='UTF-8'>)
  * `-v, --version`: show version of this script and exit

### Example

```bash
bicleaner-classify  \
        corpus.en-es.raw  \
        corpus.en-es.classifed  \
        training.en-es.yaml 
```

This will read the "`corpus.en-es.raw`" file, 
classify it with the classifier indicated in the "`training.en-es.yaml`" metadata file,
writing the result of the classification in the "`corpus.en-es.classified`" file.
Each line of the new file will contain the same content as the input file, adding a column with the score given by the Bicleaner classifier.

### Automatic test

We included a small test corpus and a script to check that your Bicleaner classifier is working as expected. 
In order to use it, just run:

```bash
python3.7 -m pytest -s tests/bicleaner_test.py
```

This will download the required language pack, classify the provided test corpus, and check the resulting classification scores. If everything went as expected, the output will be "1 passed in XX.XX seconds". All downloaded data will be removed at the end of the testing session.

## Training classifiers

In case you need to train a new classifier (i.e. because it is not available in the language packs provided at [bicleaner-data](https://github.com/bitextor/bicleaner-data/releases/latest)), you can use `bicleaner-train` .
`bicleaner-train` is a Python3 tool that allows you to train a classifier which predicts 
whether a pair of sentences are mutual translations or not and discards too noisy sentence pairs. Visit our [Wiki](https://github.com/bitextor/bicleaner/wiki/How-to-train-your-Bicleaner) for a detailed example on Bicleaner training.

### Requirements 

In order to train a new classifier, you must provide:
* A clean parallel corpus (100k pairs of sentences is the recommended size).
* SL-to-TL and TL-to-SL gzipped probabilistic bilingual dictionaries. You can check their format by downloading any of the available language packs.
   * The SL-to-TL probabilistic bilingual dictionary must contain one entry per line. Each entry must contain the following 3 fields, split by space, in this order: TL word, SL word, probability.
   * The TL-to-SL probabilistic bilingual dictionary must contain one entry per line. Each entry must contain the following 3 fields, split by space, in this order: SL word, TL word, probability.
   * We recommend filtering out entries with a very low probability: removing those with a probability 10 times lower than the maximum translation probability for each word speeds up the process and does not decrease accuracy.
   * Prior to inferring the probabilistic dictionaries, sentences must be tokenizer with the Moses tokenizer (with the `-a` flag) and lowercased.
   * You can uses Moses and MGIZA++ to obtain probabilistic dictionaries from a parallel corpus.
   * Please note that both target and source words in probabilistic bilingual dictionaries must be single words. 
* Gzipped lists of monolingual word frequencies. You can check their format by downloading any of the available language packs.
   * The SL list of word frequencies with one entry per line. Each entry must contain the following 2 fields, split by space, in this order: word frequency (number of times a word appears in text), SL word.
   * The TL list of word frequencies with one entry per line. Each entry must contain the following 2 fields, split by space, in this order: word frequency (number of times a word appears in text), TL word.
   * These lists can easily be obtained from a monolingual corpus (i.e. newscrawl or the same text used to train probabilistic bilingual dictionaries) and a command line in bash:
   
```bash
$ cat monolingual.SL \
    | sacremoses -l SL tokenize -x \
    | awk '{print tolower($0)}' \
    | tr ' ' '\n' \
    | LC_ALL=C sort | uniq -c \
    | LC_ALL=C sort -nr \
    | grep -v '[[:space:]]*1' \
    | gzip > wordfreq-SL.gz
$ cat monolingual.TL \
    | sacremoses -l TL tokenize -x \
    | awk '{print tolower($0)}' \
    | tr ' ' '\n' \
    | LC_ALL=C sort | uniq -c \
    | LC_ALL=C sort -nr \
    | grep -v '[[:space:]]*1' \
    | gzip > wordfreq-TL.gz

```
Optionally, if you want the classifier to include a porn filter, you must also provide:
* File with training dataset for porn removal classifier. Each sentence must contain at the beginning the `__label__negative` or `__label__positive` according to FastText convention. It should be lowercased and tokenized.

Optionally, if you want the classifier to include an improved fluency filter based on language models, you must also provide:
* A monolingual corpus made ONLY of noisy sentences in the SL (100k sentences is the recommended size)
* A monolingual corpus made ONLY of noisy sentences in the TL (100k sentences is the recommended size)

If not provided, since Bicleaner `0.13`, noisy corpora is produced synthetically from the training corpus.

Moreover, **`lmplz`, the command to train a KenLM language model must be in `PATH`**. See https://github.com/kpu/kenlm for instructions about its compilation and installation.

In principle, if you want to use Bicleaner to clean a partially noisy corpus, it could be difficult to find a corpus made solely of noisy sentences. Fortunately, there are two options available with Bicleaner: 

### Extracting noisy sentences from an existing corpus with heuristic rules

Given a parallel corpus, you use `bicleaner-hardrules` to extract some of its noisiest sentences using heuristic rules by running the following command:

```bash
  bicleaner-hardrules [-h]
                      [--annotated_output]
                      -s SOURCE_LANG 
                      -t TARGET_LANG
                      [--tmp_dir TMP_DIR]
                      [-b BLOCK_SIZE]
                      [-p PROCESSES]
                      [--disable_lang_ident]
                      [--disable_minimal_length]
                      [--header]
                      [--scol SCOL]
                      [--tcol TCOL]
                      [--disable_lm_filter] 
                      [--disable_porn_removal]
                      [--metadata METADATA]
                      [--lm_threshold LM_THRESHOLD]
                      [-q] 
                      [--debug]
                      [--logfile LOGFILE]
                      [input]
                      [output]
```

where `INPUT_FILE` contains a sentence-aligned parallel corpus, with a sentence pair per line. Sentences are split by tab. `OUTPUT_FILE` will contain all the input sentences, with an extra score column with `0` (if the sentence is noisy and should be discarded) or `1` (if the sentence is ok). When the `--annotated_output` flag is in use, `OUTPUT_FILE` will contain another extra column, specifying the heuristic rule applied to decide discarding each sentence (or `keep`, if the sentence is ok and should not be discarded). If the `--disable_lang_ident` flag is in use, rules that require language identification are not used. '--scol' and '--tcol' allow to indicate which columns contains source and target in the input file (default: `1`and `2`, respectively).

In order to use the LM filtering and/or porn removal, you must provide the `--metadata` (it is: the .yaml file generated by Bicleaner training).
To disable LM filtering and/or porn removal, just use the `--disable_lm_filter` and/or `--disable_porn_removal` flags.

You can then obtain the monolingual noisy corpora by "cutting" the appropriate columns (after running `bicleaner-hardrules` with the `--annotated_output` flag). Asuming scol=1 and tcol=2, and no more columns in the input corpus (so the hardrules score is the 3rd column in the output):

```bash
cat OUTPUT_FILE | awk -F'\t' '{if ($3 == 0) print $1 }' > MONOLINGUAL_NOISY.SOURCE_LANG
cat OUTPUT_FILE | awk -F'\t' '{if ($3 == 0) print $2 }' > MONOLINGUAL_NOISY.TARGET_LANG
```

### Building synthetic noisy sentences

```bash
cat TRAINING_CORPUS | cut -f1 | python3.7 bicleaner/utils/shuffle.py - > MONOLINGUAL_NOISY.SOURCE_LANG
cat TRAINING_CORPUS | cut -f2 | python3.7 bicleaner/utils/shuffle.py - > MONOLINGUAL_NOISY.TARGET_LANG
```

Since `0.13`, if no noisy corpora is provided, it's produced by Bicleaner training itself, so it has become an optional parameter.

### Parameters

It can be used as follows. Note that the parameters `--noisy_examples_file_sl`, `--noisy_examples_file_tl`, `--lm_file_sl`, `--lm_file_tl`, are mandatory if you want to enable improved fluency filter based on language models (recommended).


```bash
bicleaner_train.py [-h]
    -m METADATA
    -c CLASSIFIER
    -s SOURCE_LANG
    -t TARGET_LANG
    -d SOURCE_DICTIONARY
    -D TARGET_DICTIONARY
    -f SOURCE_WORD_FREQS
    -F TARGET_WORD_FREQS
    [-S SOURCE_TOKENIZER_COMMAND]
    [-T TARGET_TOKENIZER_COMMAND]
    [--normalize_by_length]
    [--treat_oovs]
    [--qmax_limit QMAX_LIMIT]
    [--disable_features_quest]
    [--classifier_type {mlp,svm,nn,nn1,adaboost,random_forest,extra_trees}]
    [--dump_features DUMP_FEATURES]
    [-b BLOCK_SIZE]
    [-p PROCESSES]
    [--wrong_examples_file WRONG_EXAMPLES_FILE]
    [--features_version FEATURES_VERSION]
    [--disable_lang_ident]
    [--seed SEED]
    [--relative_paths]
    [--noisy_examples_file_sl NOISY_EXAMPLES_FILE_SL]
    [--noisy_examples_file_tl NOISY_EXAMPLES_FILE_TL]
    [--lm_dev_size LM_DEV_SIZE]
    [--lm_file_sl LM_FILE_SL]
    [--lm_file_tl LM_FILE_TL]
    [--lm_training_file_sl LM_TRAINING_FILE_SL]
    [--lm_training_file_tl LM_TRAINING_FILE_TL]
    [--lm_clean_examples_file_sl LM_CLEAN_EXAMPLES_FILE_SL]
    [--lm_clean_examples_file_tl LM_CLEAN_EXAMPLES_FILE_TL]
    [--porn_removal_train PORN_REMOVAL_TRAIN]
    [--porn_removal_test PORN_REMOVAL_TEST]
    [--porn_removal_file PORN_REMOVAL_FILE]
    [--porn_removal_side {sl,tl}]
    [-q] [--debug] [--logfile LOGFILE]
    [input]

```

* positional arguments:
  * `input`: Tab-separated bilingual input file (default: Standard input)(line format: SOURCE_SENTENCE TARGET_SENTENCE, tab-separated)
* optional arguments:
  * `-h, --help`: show this help message and exit
* Mandatory:
  * `-m METADATA, --metadata METADATA`: Output training metadata (YAML file) that will be created after training.
  * `-c CLASSIFIER, --classifier CLASSIFIER`: Classifier data file that will be created after training.
  * `-s SOURCE_LANG, --source_lang SOURCE_LANG`: Source language code
  * `-t TARGET_LANG, --target_lang TARGET_LANG`: Target language code
  * `-d SOURCE_TO_TARGET_DICTIONARY, --source_dictionary SOURCE_TO_TARGET_DICTIONARY`: SL-to-TL gzipped probabilistic dictionary
  * `-D TARGET_TO_SOURCE_DICTIONARY, --target_dictionary TARGET_TO_SOURCE_DICTIONARY`: TL-to-SL gzipped probabilistic dictionary
  * `-f SOURCE_WORD_FREQ_DICTIONARY, --source_word_freqs SOURCE_WORD_FREQ_DICTIONARY`: SL gzipped word frequencies dictionary
  * `-F TARGET_WORD_FREQ_DICTIONARY, --target_word_freqs TARGET_WORD_FREQ_DICTIONARY`: TL gzipped word frequencies dictionary
* Options:
  * `-S SOURCE_TOKENIZER_COMMAND`: Source language tokenizer full command (including flags if needed). If not given, Sacremoses tokenizer is used (with `escape=False` option).
  * `-T TARGET_TOKENIZER_COMMAND`: Target language tokenizer full command (including flags if needed). If not given, Sacremoses tokenizer is used (with `escape=False` option).
  * `--normalize_by_length`: Normalize by length in qmax dict feature 
  * `--treat_oovs`: Special treatment for OOVs in qmax dict feature
  * `--qmax_limit`: Number of max target words to be taken into account, sorted by length (default: 20)
  * `--disable_features_quest`: Disable less important features 
  * `--classifier_type {svm,nn,nn1,adaboost,random_forest,extra_trees}`: Classifier type (default: extra_trees)
  * `--dump_features DUMP_FEATURES`: Dump training features to file (default: None)
  * `-b BLOCK_SIZE, --block_size BLOCK_SIZE`: Sentence pairs per block (default: 10000)
  * `-p PROCESSES, --processes PROCESSES`: Number of process to use (default: all CPUs minus one)
  * `--wrong_examples_file WRONG_EXAMPLES_FILE`: File with wrong examples extracted to replace the synthetic examples from method used by default (default: None)
  * `--features_version FEATURES_VERSION`: Version of the feature (default: extracted from the features.py file)
  * `--disable_lang_ident`: Don't apply features that use language detecting (default: False). Useful when the language in use is too similar to other languages, making the automatic identification of language not realiable.
  * `--relative_paths`: Ask training to save model files by relative path if they are in the same directory as metadata. Useful if you are going to train distributable models. (default: False)
  * `--noisy_examples_file_sl NOISY_EXAMPLES_FILE_SL`: File with noisy text in the SL. These are used to estimate the perplexity of noisy text. (Optional)
  * `--noisy_examples_file_tl NOISY_EXAMPLES_FILE_TL`: File with noisy text in the TL. These are used to estimate the perplexity of noisy text. (Optional)
  * `--lm_dev_size SIZE`: Number of sentences to be removed from clean text before training LMs. These are used to estimate the perplexity of clean text. (default: 2000)
  * `--lm_file_sl LM_FILE_SL`: Output file with the created SL language model. This file should be placed in the same directory as the YAML training metadata, as they are usually distributed together.
  * `--lm_file_tl LM_FILE_TL`: Output file with the created TL language model. This file should be placed in the same directory as the YAML training metadata, as they are usually distributed together.
  * `--lm_training_file_sl LM_TRAINING_FILE_SL`: SL text from which the SL LM is trained. If this parameter is not specified, SL LM is trained from the SL side of the input file, after removing --lm_dev_size sentences.
  * `--lm_training_file_tl LM_TRAINING_FILE_TL`: TL text from which the TL LM is trained. If this parameter is not specified, TL LM is trained from the TL side of the input file, after removing --lm_dev_size sentences.
  * `--lm_clean_examples_file_sl LM_CLEAN_EXAMPLES_FILE_SL`: File with clean text in the SL. Used to estimate the perplexity of clean text. This option must be used together with --lm_training_file_sl and both files must not have common sentences. This option replaces --lm_dev_size.
  * `--lm_clean_examples_file_tl LM_CLEAN_EXAMPLES_FILE_TL`: File with clean text in the TL. Used to estimate the perplexity of clean text. This option must be used together with --lm_training_file_tl and both files must not have common sentences. This option replaces --lm_dev_size."
  * `--porn_removal_train PORN_REMOVAL_TRAIN`: File with training dataset for porn removal classifier. Each sentence must contain at the beginning the `'__label__negative'` or `'__label__positive'` according to FastText [convention](https://fasttext.cc/docs/en/supervised-tutorial.html#getting-and-preparing-the-data). It should be lowercased and tokenized.
  * `--porn_removal_test PORN_REMOVAL_TEST`: Test set to compute precision and accuracy of the porn removal classifier.
  * `--porn_removal_file PORN_REMOVAL_FILE`: Porn removal classifier output file.
  * `--porn_removal_side {sl,tl}`: Whether the porn removal should be applied at the source or at the target language. (default: sl)
* Logging:
  * `-q, --quiet`: Silent logging mode (default: False)
  * `--debug`: Debug logging mode (default: False)
  * `--logfile LOGFILE`: Store log to a file (default: <_io.TextIOWrapper name='<stderr>' mode='w' encoding='UTF-8'>)

### Example

```bash
bicleaner-train \
          corpus.en-cs.train\
          --normalize_by_length \
          -s en \
          -t cs \
          -d dict-en-cs.gz \
          -D dict-cs-en.gz \
          -f wordfreqs-en.gz \
          -F wordfreqs-cs.gz \
          -c en-cs.classifier \
          --lm_training_file_sl lmtrain.en-cs.en --lm_training_file_tl lmtrain.en-cs.cs \
          --lm_file_sl model.en-cs.en  --lm_file_tl model.en-cs.cs \
          --porn_removal_train porn-removal.txt.en  --porn_removal_file porn-model.en \
          -m training.en-cs.yaml \
```

This will train an Extra Trees classifier for English-Czech using the corpus corpus.en-cs.train, the probabilistic dictionaries `dict-en-cs.gz` and `dict-cs-en.gz`, and the word frequency dictionaries `wordfreqs-en.gz` and `wordfreqs-cs.gz`.
This training will use 50000 good and 50000 bad examples.
The classifier data will be stored in `en-cs.classifier`, with the metadata in `training.en-cs.yaml`. The improved fluency language models will be `model.en-cs.en` and `model.en-cs.cs`, and the porn filter model will be `porn-model.en`.

The generated .yaml file provides the following information, that is useful to get a sense on how good or bad was the training (and is also a needed input file for classifying):

```yml
classifier: en-cs.classifier
classifier_type: extra_trees
source_lang: en
target_lang: cs
source_dictionary: dict-en-cs.gz
target_dictionary: dict-cs-en.gz
source_word_freqs: wordfreqs-en.gz
target_word_freqs: wordfreqs-cs.gz
normalize_by_length: True
qmax_limit: 40
disable_features_quest: True
good_test_histogram: [0, 7, 39, 45, 112, 172, 514, 2199, 6912, 0]
wrong_test_histogram: [14, 4548, 4551, 747, 118, 18, 3, 1, 0, 0]
precision_histogram: [0.5000000, 0.5003502, 0.6475925, 0.9181810, 0.9860683, 0.9977594, 0.9995846, 0.9998903, 1.0000000, nan]
recall_histogram: [1.0000000, 1.0000000, 0.9993000, 0.9954000, 0.9909000, 0.9797000, 0.9625000, 0.9111000, 0.6912000, 0.0000000]
accuracy_histogram: [0.5000000, 0.5007000, 0.7277500, 0.9533500, 0.9884500, 0.9887500, 0.9810500, 0.9555000, 0.8456000, 0.5000000]
length_ratio: 1.0111087
features_version: 4
source_lm: model.en-cs.en
target_lm: model.en-cs.cs
lm_type: CHARACTER
clean_mean_perp: -1.0744755342473238
clean_stddev_perp: 0.18368996884800565
noisy_mean_perp: -3.655791900929066
noisy_stddev_perp: 0.9989343799121657
disable_lang_ident: False
porn_removal_file: porn-model.en
porn_removal_side: sl

```

## Lite version

Although `bicleaner-train` and `bicleaner-classify` make use of parallelization by distributing workload to the available cores, some users might prefer to implement their own parallelization strategies. For that reason, single-thread version of Bicleaner classifier script is provided: `bicleaner-classify-lite`. The usage is exactly the same as for the full version, but omitting the blocksize (-b) and processes (-p) parameter.
**Note**: `bicleaner-train-lite` was removed due to the lack of usage by the users and to avoid code duplication.

___

![Connecting Europe Facility](https://www.paracrawl.eu/images/logo_en_cef273x39.png)

All documents and software contained in this repository reflect only the authors' view. The Innovation and Networks Executive Agency of the European Union is not responsible for any use that may be made of the information it contains.
