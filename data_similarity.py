import numpy as np
import scipy.spatial.distance as dist
import os

def data_similarity(person_id = 'F0001', in_dir = 'prediction/'):
    print(person_id)
    train_dir = 'classifier_'+person_id+'/neutral/'
    trainls = os.listdir(train_dir)
    test_dir = 'classifier_'+person_id+'/feature_sample_youtube/'
    testls = os.listdir(test_dir)

    test_label = np.load('/root/classifier/sample_youtube/test_label.npy')
    class_label = np.load(in_dir+person_id+'.npy')

    test_data = []
    for i in range(len(test_label)):
        if test_label[i] >= 0:
                test_data.append(np.load(test_dir + testls[i]))
    ne_test = []
    for i in range(len(class_label)):
        if class_label[i] == 0:
                ne_test.append(test_data[i])
    
    train_data = []
    for ne in trainls:
        train_data.append(np.load(train_dir + ne))

    ne_train = np.array(train_data)
    ne_test = np.array(ne_test)

    data_distance = dist.directed_hausdorff(ne_train, ne_test)[0]

    return test_data, class_label, data_distance
