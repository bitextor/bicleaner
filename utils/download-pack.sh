#!/bin/bash

usage() {
    echo "Script to download Bicleaner language packs."
    echo "It will try to download lang1-lang2.tar.gz and if it does not exist it will try lang2-lang1.tar.gz ."
    echo
    echo "Usage: `basename $0` <lang1> <lang2> <download_path>"
    echo "      <lang1>         Language 1."
    echo "      <lang2>         Language 2."
    echo "      <download_path> Path where downloaded language pack should be placed."
}

invalid_url(){
    wget -S --spider -o - $1 | grep -q '404 Not Found'
}

if [[ $# -lt 2 ]]
then
    echo "Wrong number of arguments: $@" 2>&1
    usage 2>&1
    exit 1
fi

URL="https://github.com/bitextor/bicleaner-data/releases/latest/download"
L1=$1
L2=$2
if [ "$3" != "" ]; then
    DOWNLOAD_PATH=$3
else
    DOWNLOAD_PATH="."
fi

if invalid_url $URL/$L1-$L2.tar.gz
then
    >&2 echo $L1-$L2 language pack does not exist, trying $L2-$L1...
    if invalid_url $URL/$L2-$L1.tar.gz
    then
        >&2 echo $L2-$L1 language pack does not exist
    else
        wget -P $DOWNLOAD_PATH $URL/$L2-$L1.tar.gz
        tar xvf $DOWNLOAD_PATH/$L2-$L1.tar.gz -C $DOWNLOAD_PATH
        rm $DOWNLOAD_PATH/$L2-$L1.tar.gz
    fi
else
    wget -P $DOWNLOAD_PATH $URL/$L1-$L2.tar.gz
    tar xvf $DOWNLOAD_PATH/$L1-$L2.tar.gz -C $DOWNLOAD_PATH
    rm $DOWNLOAD_PATH/$L1-$L2.tar.gz
fi

echo Finished
