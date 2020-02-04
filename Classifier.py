import os, cv2, sys
import numpy as np
from data_similarity import data_similarity
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('-ts', '--similarity', default='0', help='Threshold of similarity.')
parser.add_argument('-to', '--outlier', default='05', help='Threshold of anomaly detection.')
parser.add_argument('-f', '--feature', default='feature_sample_youtube/', help='Feature directory of test images.')
parser.add_argument('-in', '--in_name', default='sample_youtube/', help='Input movie name.')
parser.add_argument('-m', '--movie', default='/root/classifier/sample_youtube/movie/', help='Input image directory of test.')
parser.add_argument('-l', '--label', help='Labels of neutral or not.')
args = parser.parse_args()

sim_th = args.similarity
out_th = args.outlier
feat_dir = args.feature
#feature_sample_youtube/
in_name = args.in_name
#sample_youtube/
in_dir = in_name + 'prediction_' + out_th + '/'
#sample_youtube/prediction_05/
out_dir = in_dir + 'result_' + sim_th + '/'
#sample_youtube/prediction_05/result_0/
movie = args.movie
true_label = args.label
sim_th = float('0.'+sim_th)


featls = []
labells = []
distls = []
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

for s in ['F', 'M']:
    for i in range(1, 21):
        person_id = s + str(i).zfill(4)
        feat, label, dist = data_similarity(person_id, in_dir, feat_dir)
        
        featls.append(feat) #classifier*208
        labells.append(label) #classifier*frame
        distls.append(dist) #classifier*1

distls = np.array(distls)
labells = np.array(labells)

distls = distls / max(distls)
distls = np.abs(distls - 1)

for d in distls:
    print(d)
ne_ind = np.where(distls>=sim_th)[0]

if len(ne_ind)==0:
    add_val = max(distls[np.where(distls<=sim_th)[0]])
    add_ind = np.where(distls == add_val)
    ne_ind = np.insert(ne_ind, 0, add_ind)
    ne_ind.sort()

result = np.zeros((len(labells[0])))
for i in ne_ind:
    print(i, distls[i])
    result = result + np.array(labells[i])*distls[i]

result = result / max(result)

pred = []
th = 0.5
#th = len(ne_ind)/2
for i in range(len(result)):
    if result[i] < th:
        print(str(i).zfill(3), ' : 0     ', str(result[i]))
        pred.append(0)
    elif result[i] > th:
        print(str(i).zfill(3), ' : 1     ', str(result[i]))
        pred.append(1)
    else:
        print(str(i).zfill(3), ' : equal ', str(result[i]))
        pred.append(1)

np.save(out_dir + 'result', np.array(pred))

imgls = [x for x in os.listdir(movie) if not x.startswith('.')]
imgls.sort()
i=0
for img in imgls:
    frame = cv2.imread(movie + img)
    height, width, _ = frame.shape
    height = int(height)
    width = int(width)
    if pred[i]>0:
        c = (0, 0, 255)
        txt = 'EXP'
    else:
        c = (255, 0, 0)
        txt = 'NE'
    cv2.rectangle(frame, (0, int(height*9/10)), (width, height), c, thickness=-1)
    cv2.putText(frame, txt, (0, height-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
    cv2.imwrite(out_dir+str(i).zfill(3)+'.png', frame)
    i+=1

#evaluate RECALL etc
if true_label != None:
    label = np.load(true_label)
    #'/root/classifier/sample_youtube/test_label.npy'

    ind_zero_label = [i for i, x in enumerate(label) if x == 0]
    ind_zero_pred = [i for i, x in enumerate(pred) if x == 0]
    ind_one_label = [i for i, x in enumerate(label) if x == 1]
    ind_one_pred = [i for i, x in enumerate(pred) if x == 1]

    TP = list(set(ind_zero_label) & set(ind_zero_pred))
    TN = list(set(ind_one_label) & set(ind_one_pred))
    FP = list(set(ind_one_label) & set(ind_zero_pred))
    FN = list(set(ind_zero_label) & set(ind_one_pred)) 

    FNR = len(FN) / (len(TP) + len(FN))
    TNR = len(TN) / (len(TN) + len(FP))

    f = open(out_dir + 'result.txt', 'w')
    f.write('selected encoder = ' + str(len(ne_ind)) + '\n')
    for ind in ne_ind:
        if ind < 20:
            f.write(str('F') + str(ind+1) + ', ')
        if ind >= 20:
            f.write(str('M') + str(ind-19) + ', ')

    f.write('\n\n')

    f.write('TP = ' + str(len(TP)) + '\n')
    f.write('FN = ' + str(len(FN)) + '\n')
    f.write('FP = ' + str(len(FP)) + '\n')
    f.write('TN = ' + str(len(TN)) + '\n\n')

    f.write('False Negative Rate = ' + str(FNR)[:5] + '\n')
    f.write('True Negative Rate  = ' + str(TNR)[:5])
    f.close()
