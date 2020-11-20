import numpy as np
import logging

try:
    from .features import feature_extract
    from .bicleaner_hardrules import wrong_tu
except (ImportError, SystemError):
    from features import feature_extract
    from bicleaner_hardrules import wrong_tu

# Classify sentences from input and place them at output
# that can be either files or stdin/stdout
def classify(args, input, output, lm_filter, source_tokenizer, target_tokenizer, porn_tokenizer):
    nline = 0
    buf_sent = []
    buf_sent_sl = []
    buf_sent_tl = []
    buf_score = []

    # Read from input file/stdin
    for line in input:
        nline += 1
        parts = line.split("\t")

        # Parse fields and buffer sentences
        sl_sentence=None
        tl_sentence=None
        if len(parts) >= max(args.scol, args.tcol):
            sl_sentence=parts[args.scol -1].strip()
            tl_sentence=parts[args.tcol -1].strip()
        else:
            logging.error("ERROR: scol ({}) or tcol ({}) indexes above column number ({}) on line {}".format(args.scol, args.tcol, len(parts), nline))

        buf_sent.append(line)

        # Buffer sentences that are not empty and pass hardrules
        if sl_sentence and tl_sentence and (args.disable_hardrules or wrong_tu(sl_sentence,tl_sentence, args, lm_filter, args.porn_removal, porn_tokenizer)== False):
            buf_score.append(1)
            buf_sent_sl.append(sl_sentence)
            buf_sent_tl.append(tl_sentence)
        else:
            buf_score.append(0)

        # Score batch and empty buffers
        if (nline % args.block_size) == 0:
            classify_batch(args, output, buf_sent, buf_sent_sl, buf_sent_tl, buf_score, source_tokenizer, target_tokenizer)
            buf_sent = []
            buf_sent_sl = []
            buf_sent_tl = []
            buf_score = []

    # Score remaining sentences
    if len(buf_sent) > 0:
        classify_batch(args, output, buf_sent, buf_sent_sl, buf_sent_tl, buf_score, source_tokenizer, target_tokenizer)

    return nline

# Score a batch of sentences
def classify_batch(args, output, buf_sent, buf_sent_sl, buf_sent_tl, buf_score, source_tokenizer, target_tokenizer):
    # Tokenize
    buf_tok_sl = source_tokenizer.tokenize(buf_sent_sl)
    buf_tok_tl = target_tokenizer.tokenize(buf_sent_tl)

    # Compute features
    buf_feat = []
    for sl_sent, tl_sent, sl_sent_t, tl_sent_t in zip(buf_sent_sl, buf_sent_tl, buf_tok_sl, buf_tok_tl):
        features = feature_extract(sl_sent, tl_sent, sl_sent_t, tl_sent_t, args)
        buf_feat.append([float(v) for v in features])

    # Classifier predictions
    predictions = args.clf.predict_proba(np.array(buf_feat)) if len(buf_feat) > 0 else []
    p = iter(predictions)

    # Print sentences and scores to output
    for score, sent in zip(buf_score, buf_sent):
        if score == 1:
            if args.score_only:
                output.write("{0:.3f}".format((next(p)[1])))
            else:
                output.write(sent.strip())
                output.write("\t")
                output.write("{0:.3f}".format((next(p)[1])))
            output.write("\n")
        else:
            if args.score_only:
                output.write("0")
            else:
                output.write(sent.rstrip("\n"))
                output.write("\t0")
            output.write("\n")
