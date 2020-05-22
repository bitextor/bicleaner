Bicleaner 0.14: 
* Bicleaner hardrules changes:
  * New rule: filter out porn videos.

Bicleaner 0.13:
* Bicleaner hardrules changes:
  * Rule change: Relaxed c_minimal_length to accept 3-word sentences	
  * New feature: LM filtering (moved from Bicleaner Classify)
  * New parameter: `--disable_lm_filter`, `--metadata` and `--lm_threshold`, to support LM filtering
* Bicleaner training changes: 
  * New parameter: Features relying on language identification can be disabled with flag `--disable_lang_ident` (this will be outputed in the .yaml file and used by Bicleaner clasifier)
  * New feature: Debug mode now gives information on random forest feature importances
  * Parameter change: --noisy_examples_file_sl and --noisy_examples_file_tl are now optional
  * Parameter change: input now must be more than 10K sentences long
  * Removed INFO messages when processes starting/ending (except when debugging)
* Bicleaner classifier changes:
  * `--disable_lang_ident` flag is now read from the .yaml file
  * Removed feature: LM filtering (moved to Bicleaner Hardrules)
  * New parameter: `--disable_lm_filter`
  * Removed parameters: `--keep_lm_result`,  `--threshold`
* Other:
  * Updated requirements
  
  
  
Bicleaner 0.12:
* Bicleaner hardrules changes:
  * New rule: c_identical_wo_punct to reject sentences only different in punctuation (and it's case insensitive)
  * New rule:  Sentences containing "Re:" are rejected
  * Rule change: c_minimal_length now rejects sentences with both sides <= 3 words (instead of only one)
  * Rule change: c_identical and c_identical_wo_digits now is case insensitive
  * Rule change: Breadcrumbs rule now split into c_no_breadcrumbs1 and c_no_breadcrumbs2
  * Rule change: Breadcrumbs2 now includes character "·" in the rejected characters
  * Rule change: c_length now compares byte length ratio (will avoid rejecting valid sentences due to length ratio when comparing languages with different alphabets)
  * Changed behaviour for `--annotated_output` argument in hardrules. See README.md for more information.
  * New parameter: `--disable_lang_ident` flag to avoid applying rules that need to identify the language
* Bicleaner classify changes:  
  * Now using only 3 decimal places for Bicleaner score and LM score
  * Removed INFO messages when processes starting/ending (except when debugging)
  * New parameter: '--disable_hardrules' flag to avoid applying hardrules
  * New parameter: '--disable_lang_ident' flag to avoid applying rules that need to identify the language
  * New parameter: '--score_only' flag to output only Bicleaner scores (proposed by [@kirefu](https://github.com/kirefu))
* Bicleaner features changes:
  * Fixed bug when probability in prob_dict is 0 (issue [#19](https://github.com/bitextor/bicleaner/issues/19))
* Other:
  * Fixed sklearn version to 0.19.1
  * Added utilities for training: `shuffle.py` and `dict_pruner.py`
  * Updated instalation guides in readme
