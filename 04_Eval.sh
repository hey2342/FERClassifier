#!/bin/bash

python AnomalyDetection.py 0.3 prediction_3/
python Classifier.py 0.25 prediction_3/ prediction_3/result_25/
