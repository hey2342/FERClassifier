import numpy as np
import scipy.spatial.distance as dist
import os

def data_similarity(person_id = 'F0001', in_dir = 'sample/youtube/prediction/', feat_dir = 'feature_sample_youtube/'):
    print(person_id)
    #read train neutral feature
    train_dir = 'classifier_'+person_id+'/neutral/'
    trainls = os.listdir(train_dir)
    trainls.sort()

    ne_train = []
    for name in trainls:
        ne_train.append(np.load(train_dir + name))

    #read test neutral feature
    test_dir = 'classifier_'+person_id+'/'+ feat_dir
    testls = os.listdir(test_dir)
    testls.sort()
    class_label = np.load(in_dir+person_id+'.npy')

    ne_test = []
    test_data = []
    for i in range(len(testls)):
        if class_label[i] == 0:
                ne_test.append(np.load(test_dir + testls[i]))
        test_data.append(np.load(test_dir + testls[i]))


    ne_train = np.array(ne_train)
    ne_test = np.array(ne_test)

    data_distance = dist.directed_hausdorff(ne_train, ne_test)[0]

    return test_data, class_label, data_distance
