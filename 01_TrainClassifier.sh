#!/bin/bash

for i in `seq 1 20`
do
    fid="F"`printf %04d $i`
    mid="M"`printf %04d $i`

    cp -r encoder_base encoder_$fid    
    python Encoder.py train ListImages/Train_$fid.txt -c encoder_$fid/ --finetuning

    cp -r encoder_base encoder_$mid
    python Encoder.py train ListImages/Train_$mid.txt -c encoder_$mid/ --finetuning

done

