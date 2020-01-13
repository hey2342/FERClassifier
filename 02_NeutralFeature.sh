#!/bin/bash

for i in `seq 1 20`
do
    fid="F"`printf %04d $i`
    mid="M"`printf %04d $i`

    python Encoder.py test ListNeutral/Test_$fid.txt -c classifier_$fid/ -f classifier_$fid/neutral/

    python Encoder.py test ListNeutral/Test_$mid.txt -c classifier_$mid/ -f classifier_$mid/neutral/

done

