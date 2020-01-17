import os, cv2, sys
import numpy as np
from data_similarity import data_similarity

featls = []
labells = []
distls = []

sim_th = float(sys.argv[1]) #0.25 0.5 0.75
in_dir = sys.argv[2]
out_dir = sys.argv[3] #sample_youtube/result/
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

for s in ['F', 'M']:
    for i in range(1, 21):
        person_id = s + str(i).zfill(4)
        feat, label, dist = data_similarity(person_id, in_dir)
        
        featls.append(feat) #classifier*208
        labells.append(label) #classifier*frame
        distls.append(dist) #classifier*1

distls = np.array(distls)

distls = distls / max(distls)
distls = np.abs(distls - 1)

for d in distls:
    print(d)
ne_ind = np.where(distls>sim_th)[0]

if len(ne_ind)%2==0:
    add_val = max(distls[np.where(distls<=sim_th)[0]])
    add_ind = np.where(distls == add_val)
    ne_ind = np.insert(ne_ind, 0, add_ind)
    ne_ind.sort()

result = []
for i in ne_ind:
    print(i)
    result.append(labells[i])

result_sum = np.sum(result, 0)
pred = []
th = len(ne_ind)/2
for i in range(len(result_sum)):
    if result_sum[i] < th:
        print(str(i).zfill(3), ' : 0')
        pred.append(0)
    elif result_sum[i] > th:
        print(str(i).zfill(3), ' : 1')
        pred.append(1)
    else:
        print(str(i).zfill(3), ' : 0=')
        pred.append(1)

cap = cv2.VideoCapture('/root/classifier/sample_youtube/sample_youtube.mp4')
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

label = np.load('/root/classifier/sample_youtube/test_label.npy')

j=0
i=0
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        if label[j] >= 0:
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
        j+=1
    else:
        break

cap.release()

label = [x for x in label if x>0]

ind_zero_label = [i for i, x in enumerate(label) if x == 0]
ind_zero_pred = [i for i, x in enumerate(pred) if x == 0]
zero_tp = list(set(ind_zero_label) & set(ind_zero_pred))

ind_one_label = [i for i, x in enumerate(label) if x == 1]
ind_one_pred = [i for i, x in enumerate(pred) if x == 0]
one_tp = list(set(ind_one_label) & set(ind_one_pred))

zero_precision = len(zero_tp) / len(ind_zero_pred)
one_precision = len(one_tp) / len(ind_one_pred)
zero_recall = len(zero_tp) / len(ind_zero_label)
one_recall =  len(one_tp) / len(ind_one_label)

f = open(out_dir + 'result.txt', 'w')
f.write('selected encoder = ', ne_ind)

f.write('neutral precision    = ' + str(zero_precision)[:5] + ‘\n’)
f.write('expression precision = ' + str(one_precision)[:5] + ‘\n’ + ‘\n’)

f.write('neutral recall    = ' + str(zero_recall)[:5] + ‘\n’)
f.write('expression recall = ' + str(one_recall)[:5])
f.close()
np.save(out_dir + 'result', np.array(pred))

