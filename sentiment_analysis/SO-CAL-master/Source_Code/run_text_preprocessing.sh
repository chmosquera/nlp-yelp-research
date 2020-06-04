#!/bin/bash

SAMPLES="../../samples/*"
for file in $SAMPLES
do
    python text_preprocessing/preprocess.py -i $file -o './analyzed/' -a 'tokenize,ssplit,pos'
done
