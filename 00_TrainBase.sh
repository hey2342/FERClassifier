#!/bin/bash

python Encoder.py train ListImages/Train_001.txt -c classifier_base/
python Encoder.py test ListImages/Test_001.txt -c classifier_base/
