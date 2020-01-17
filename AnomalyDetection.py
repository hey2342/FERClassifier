import os, sys
import numpy as np
from sklearn.svm import OneClassSVM

th = float(sys.argv[1])
out_dir = sys.argv[2]
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

labels = np.load('/root/classifier/sample_youtube/test_label.npy')
#labels = np.load('./root/classifier/sample_youtube/sample_youtube_label.npy')


for s in ['F', 'M']:
    for i in range(1, 21):
        person_id = s+str(i).zfill(4)

        feat_dir = './classifier_'+person_id+'/feature_sample_youtube/'
        print('load : '+feat_dir)
        featls = os.listdir(feat_dir)
        featls.sort()

        featnp = []
        for j, feat in enumerate(featls):
            if labels[j]>=0:
                featnp.append(np.load(feat_dir + feat))
        
        ocsvm = OneClassSVM(nu=th, kernel='rbf', gamma="auto")
        #outlier=-1,  inlier=1
        ocsvm.fit(featnp)
        preds = ocsvm.predict(featnp)
        preds = (preds+1)*0.5
        #outlier=0, inlier=1
        preds = np.abs(preds-1)
        #outlier=1(exp), inlier=0(ne)

        np.save(out_dir + person_id, preds)
