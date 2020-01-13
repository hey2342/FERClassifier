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
result = []
th = len(ne_ind)/2
ne = 0
exp = 0
for i in range(len(result_sum)):
    if result_sum[i] < th:
        print(str(i).zfill(3), ' : 0')
        ne+=1
        result.append(0)
    elif result_sum[i] > th:
        print(str(i).zfill(3), ' : 1')
        exp+=1
        result.append(1)
    else:
        print(str(i).zfill(3), ' : 0=')
        exp+=1
        result.append(1)

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
            if result[i]>0:
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
        
one_teach = np.count_nonzero(label>0)
zero_teach = np.count_nonzero(label==0)

one_pred = exp / one_teach
zero_pred = ne / zero_teach

ne_result = 'neutral    : ' + str(ne) + ' / ' + str(zero_teach) + ' = ' + str(zero_pred)[:5]
exp_result = 'expression : ' + str(exp) + ' / ' + str(one_teach) + ' = ' + str(one_pred)[:5]

f = open(out_dir + 'result.txt', 'w')
f.write(ne_result + '\n')
f.write(exp_result)
