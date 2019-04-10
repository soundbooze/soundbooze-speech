#!/bin/sh
#for i in 1 2 3 4 5
while [ 1 ]
do
  rm -f test/test.csv test/out.wav
  rm -rf cache/
  jack_capture -d 4 test/out.wav > /dev/null 2>&1 
  Rscript xtract.R 2> /dev/null
  python voice.py 2> /dev/null
done
