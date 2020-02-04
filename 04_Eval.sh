#!/bin/bash


#for i in `seq -w 5 5 50`
#do
    #python AnomalyDetection.py $i sample_youtube/
#    python Classifier.py -ts 0 -to $i -f feature_sample_youtube/ -in sample_youtube/ -m ~/classifier/sample_youtube/movie/ -l ~/classifier/sample_youtube/test_label.npy
#done

#python AnomalyDetection.py 20 sample_youtube/
python Classifier.py -ts 0 -to 20 -f feature_sample_youtube/ -in sample_youtube/ -m ~/classifier/sample_youtube/movie/ -l ~/classifier/sample_youtube/test_label.npy
