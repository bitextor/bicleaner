#!/usr/bin/env python

import os
import sys
import logging
import traceback
import subprocess
import joblib

from hardrules.hardrules import Hardrules
from heapq import heappush, heappop
from multiprocessing import Queue, Process, Value, cpu_count
from tempfile import NamedTemporaryFile
from timeit import default_timer


#Allows to load modules while inside or outside the package
try:
    from .classify import classify, argument_parser, load_metadata
    from .util import logging_setup
    from .tokenizer import Tokenizer
except (ImportError, SystemError):
    from classify import classify, argument_parser, load_metadata
    from util import logging_setup
    from tokenizer import Tokenizer

logging_level = 0

def initialization():
    global logging_level

    # Validating & parsing
    parser, groupO, _ = argument_parser()
    groupO.add_argument('-b', '--block_size', type=int, default=1000, help="Sentence pairs per block")
    groupO.add_argument('-p', '--processes', type=int, default=max(1, cpu_count()-1), help="Number of processes to use")
    args = parser.parse_args()

    # Set up logging
    logging_setup(args)
    logging_level = logging.getLogger().level

    # Load metadata YAML
    args = load_metadata(args, parser)

    return args


def classifier_process(i, jobs_queue, output_queue, args):
    # Load tokenizers and Hardrules object
    source_tokenizer = Tokenizer(args.source_tokenizer_command, args.source_lang)
    target_tokenizer = Tokenizer(args.target_tokenizer_command, args.target_lang)
    hardrules = Hardrules(args)

    # If there are still jobs pending
    # grab one input file and place scores at the output queue
    while True:
        job = jobs_queue.get()
        if job:
            logging.debug("Job {0}".format(job.__repr__()))
            nblock, filein_name = job
            ojob = None
            with open(filein_name, 'r') as filein, NamedTemporaryFile(mode="w", delete=False, dir=args.tmp_dir) as fileout:
                logging.debug("Classification: creating temporary filename {0}".format(fileout.name))

                # Score sentences
                classify(args, filein, fileout, source_tokenizer, target_tokenizer, hardrules)

                ojob = (nblock, fileout.name)
                filein.close()
                fileout.close()

            if ojob:
                output_queue.put(ojob)

            os.unlink(filein_name)
        else:
            logging.debug("Exiting worker {}".format(job.__repr__()))
            break

def mapping_process(args, jobs_queue):
    logging.info("Start mapping")
    nblock = 0
    nline = 0
    mytemp = None
    for line in args.input:
        if (nline % args.block_size) == 0:
            logging.debug("Creating block {}".format(nblock))
            if mytemp:
                job = (nblock, mytemp.name)
                mytemp.close()
                jobs_queue.put(job)
                nblock += 1
            mytemp = NamedTemporaryFile(mode="w", delete=False, dir=args.tmp_dir)
            logging.debug("Mapping: creating temporary filename {0}".format(mytemp.name))
        mytemp.write(line)

        nline += 1

    if nline > 0:
        job = (nblock, mytemp.name)
        mytemp.close()        
        jobs_queue.put(job)

    return nline

def reduce_process(output_queue, args):
    h = []
    last_block = 0
    while True:
        logging.debug("Reduce: heap status {0}".format(h.__str__()))
        while len(h) > 0 and h[0][0] == last_block:
            nblock, filein_name = heappop(h)
            last_block += 1

            with open(filein_name, 'r') as filein:
                for i in filein:
                    args.output.write(i)

                    if args.discarded_tus:
                        args.discarded_tus.write(i)
                filein.close()
            os.unlink(filein_name)

        job = output_queue.get()
        if job:
            nblock, filein_name = job
            heappush(h, (nblock, filein_name))
        else:
            logging.debug("Exiting reduce loop")
            break

    if len(h) > 0:
        logging.debug("Still elements in heap")

    while len(h) > 0 and h[0][0] == last_block:
        nblock, filein_name = heapq.heappop(h)
        last_block += 1

        os.unlink(filein_name)

    if len(h) != 0:
        logging.error("The queue is not empty and it should!")

    logging.info("Classification finished. Output available in {}".format(args.output.name))
    args.output.close()
    if args.discarded_tus:
        logging.info("Discarded TUs are available in {}".format(args.discarded_tus.name))
        args.discarded_tus.close()

# Filtering input texts
def perform_classification(args):
    global logging_level
    
    time_start = default_timer()
    logging.debug("Starting process")
    logging.debug("Running {0} workers at {1} rows per block".format(args.processes, args.block_size))

    process_count = max(1, args.processes)
    maxsize = 1000 * process_count

    output_queue = Queue(maxsize = maxsize)
    worker_count = process_count

    # Start reducer
    logging.disable(logging.INFO)
    reduce = Process(target = reduce_process,
                     args   = (output_queue, args))

    reduce.start()
    logging.disable(logging.DEBUG)

    # Start workers
    jobs_queue = Queue(maxsize = maxsize)
    workers = []

    for i in range(worker_count):
        filter = Process(target = classifier_process, #profile_classifier_process
                         args   = (i, jobs_queue, output_queue, args))
        filter.daemon = True # dies with the parent process
        filter.start()
        workers.append(filter)


    # Mapper process (foreground - parent)
    nline = mapping_process(args, jobs_queue)
    args.input.close()

    # Worker termination
    for _ in workers:
        jobs_queue.put(None)

    logging.info("End mapping")

    errors = False
    for w in workers:
        w.join()
        if w.exitcode != 0:
            errors = True

    # Reducer termination
    output_queue.put(None)
    reduce.join()

    # Stats
    logging.info("Finished")
    elapsed_time = default_timer() - time_start
    logging.info("Total: {0} rows".format(nline))
    logging.info("Elapsed time {0:.2f} s".format(elapsed_time))
    logging.info("Troughput: {0} rows/s".format(int((nline*1.0)/elapsed_time)))

    return errors
### END PARALLELIZATION METHODS ###

def main(args):
    logging.info("Executing main program...")
    errors = perform_classification(args)
    if errors:
        logging.error("Program finished with errors")
        sys.exit(1)
    else:
        logging.info("Program finished")

if __name__ == '__main__':
    try:
        logging_setup()
        args = initialization() # Parsing parameters
        main(args)  # Running main program
    except Exception as ex:
        tb = traceback.format_exc()
        logging.error(tb)
        sys.exit(1)
