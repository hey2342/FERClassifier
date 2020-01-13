#!/bin/bash

for i in `seq 1 20`
do
    fid="F"`printf %04d $i`
    mid="M"`printf %04d $i`

    python Encoder.py test ListImages/Test_sample_youtube.txt -c classifier_$fid/ -f classifier_$fid/feature_sample_youtube/

    python Encoder.py test ListImages/Test_sample_youtube.txt -c classifier_$mid/ -f classifier_$mid/feature_sample_youtube/

done

