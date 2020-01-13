#!/bin/bash

for i in `seq 1 20`
do
    fid="F"`printf %04d $i`
    mid="M"`printf %04d $i`

    cp -r classifier_base classifier_$fid    
    python Encoder.py train ListImages/Train_$fid.txt -c classifier_$fid/ --finetuning

    cp -r classifier_base classifier_$mid
    python Encoder.py train ListImages/Train_$mid.txt -c classifier_$mid/ --finetuning

done

