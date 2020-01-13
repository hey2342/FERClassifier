import numpy as np
import cv2
import torch
import torch.utils.data


def LoadAllDataset(data_dir, lmark_num = 68, color=1):
    f = open(data_dir, 'r')
    data_list = f.readlines()

    lmark = []
    reye = []
    leye = []
    mouth = []
    target = []
    for i, d in enumerate(data_list):
        parts = []

        name, label = d.replace('\n', '').split(' ')
        name = '/root/classifier/' + name
        lmark_dir = name.replace('data', 'np').replace('.png', '.npy')
       
        reye_dir = name.replace('data', 'reye')
        leye_dir = name.replace('data', 'leye')
        mouth_dir = name.replace('data', 'mouth')

        land = np.load(lmark_dir)[68-lmark_num:]
        lmark.append(np.reshape(land, (lmark_num*3)))
        reye.append(cv2.resize(cv2.imread(reye_dir, color), (32, 16)).transpose((2,0,1)))
        leye.append(cv2.resize(cv2.imread(leye_dir, color), (32, 16)).transpose((2,0,1)))
        mouth.append(cv2.resize(cv2.imread(mouth_dir, color), (32, 16)).transpose((2,0,1)))
            
        target.append(float(label))


    lmark = torch.Tensor(np.array(lmark))
    reye = torch.Tensor(np.array(reye))
    leye = torch.Tensor(np.array(leye))
    mouth = torch.Tensor(np.array(mouth))
    target = torch.Tensor(np.array(target))
    
    return torch.utils.data.TensorDataset(lmark, reye, leye, mouth, target)


