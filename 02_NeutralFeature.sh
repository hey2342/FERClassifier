#!/bin/bash

for i in `seq 1 20`
do
    fid="F"`printf %04d $i`
    mid="M"`printf %04d $i`

    python Encoder.py test ListNeutral/Test_$fid.txt -c encoder_$fid/ -f encoder_$fid/neutral/

    python Encoder.py test ListNeutral/Test_$mid.txt -c encoder_$mid/ -f encoder_$mid/neutral/

done

