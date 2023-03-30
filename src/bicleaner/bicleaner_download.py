#!/usr/bin/env python
from tempfile import NamedTemporaryFile
from argparse import ArgumentParser
import tarfile
import logging
import sys

from requests import get


GITHUB_URL = "https://github.com/bitextor/bicleaner-data/releases/latest/download"


def logging_setup(args):
    logger = logging.getLogger()
    logger.handlers = []
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(h)


def main():
    parser = ArgumentParser(
            description='Download Bicleaner models from GitHub.'
                        'It will try to download lang1-lang2.tar.gz'
                        ' and if it does not exist, it will try lang2-lang1.tar.gz.')
    parser.add_argument('src_lang', type=str,
                        help='Source language')
    parser.add_argument('trg_lang', type=str,
                        help='Target language')
    parser.add_argument('download_path', type=str,
                        help='Path to download the model')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    logging_setup(args)

    # Download from github
    url = f'{GITHUB_URL}/{args.src_lang}-{args.trg_lang}.tar.gz'
    logging.info(f"Trying {url}")
    response = get(url, allow_redirects=True, stream=True)
    if response.status_code == 404:
        response.close()
        logging.warning(f"{args.src_lang}-{args.trg_lang} language pack does not exist" \
                        + f" trying {args.trg_lang}-{args.src_lang}...")
        url = f'{GITHUB_URL}/{args.trg_lang}-{args.src_lang}.tar.gz'
        response = get(url, allow_redirects=True, stream=True)

        if response.status_code == 404:
            response.close()
            logging.error(f"{args.trg_lang}-{args.src_lang} language pack does not exist")
            sys.exit(1)

    # Write the tgz to temp and extract to desired path
    with NamedTemporaryFile() as temp:
        logging.info("Downloading file")
        with open(temp.name, mode='wb') as f:
            f.writelines(response.iter_content(1024))
        response.close()

        logging.info(f"Extracting tar.gz file to {args.download_path}")
        with tarfile.open(temp.name) as f:
            f.extractall(args.download_path)


if __name__ == "__main__":
    main()
