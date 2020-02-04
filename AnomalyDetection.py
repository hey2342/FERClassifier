import os, sys
import numpy as np
from sklearn.svm import OneClassSVM

th = float('0.' + sys.argv[1]) #30(anomaly rate)
test_dir = sys.argv[2] #sample_youtube/
out_dir = test_dir + 'prediction_' + sys.argv[1]+'/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

for s in ['F', 'M']:
    for i in range(1, 21):
        person_id = s+str(i).zfill(4)

        feat_dir = './classifier_'+person_id+'/feature_'+test_dir
        print('load : '+feat_dir)
        featls = os.listdir(feat_dir)
        featls.sort()

        featnp = []
        for feat in featls:
            featnp.append(np.load(feat_dir + feat))
        
        ocsvm = OneClassSVM(nu=th, kernel='rbf', gamma="auto")
        ocsvm.fit(featnp)
        preds = ocsvm.predict(featnp)
        #outlier=-1,  inlier=1
        preds = (preds+1)*0.5
        #outlier=0, inlier=1
        preds = np.abs(preds-1)
        #outlier=1(exp), inlier=0(ne)

        np.save(out_dir + person_id, preds)
