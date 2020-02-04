import os, sys, cv2
import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable

from LoadAllDataset import LoadAllDataset as LD

from evaluation import evaluation

import argparse


parser = argparse.ArgumentParser()

parser.add_argument('t', choices=['train', 'test'], help='Choice train or test.')
parser.add_argument('input', help='Input source.')
parser.add_argument('-c', '--classifier_dir', default='./classifier/', help='Output directory.')
parser.add_argument('--finetuning', action='store_true', help='Finetuning or not in case train mode.')
parser.add_argument('-f', '--feature', default='./feature/', help='Feature directory in case test mode.')
args = parser.parse_args()

num_epochs = 100
batch_size = 36
learning_rate = 0.001

LMARK_NUM = 51

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        
        #encode landmark
        self.lmark_encoder = nn.Sequential(
            nn.Linear(LMARK_NUM*3, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU())
 
        #encode reye image
        self.reye_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU())
        #encode leye image
        self.leye_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU())

        #encode mouth image
        self.mouth_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU())

        #classify
        self.classifier = nn.Sequential(
            nn.Linear(208, 208),
            nn.ReLU(),
            nn.Linear(208, 1),
            nn.Sigmoid())

    def forward(self, lmark, reye, leye, mouth):
        x1 = self.lmark_encoder(lmark)
        x2 = self.reye_encoder(reye).view(-1, 64)
        x3 = self.leye_encoder(leye).view(-1, 64)
        x4 = self.mouth_encoder(mouth).view(-1,64)
        z = torch.cat([x1, x2, x3, x4], 1)
        y = self.classifier(z)
        return y, z


def train(in_dir, class_dir, fine_tuning):
    cuda = torch.cuda.is_available()
    if cuda:
        print('cuda is available!')

    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
    
    img_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = LD(in_dir, lmark_num = LMARK_NUM, color=1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = Encoder()
    print(model)
    if cuda:
        model.cuda()
    
    if fine_tuning >0:
        print('fine tuning')
        model.load_state_dict(torch.load(class_dir+'classifier.pth'))
 
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(),
                             lr=learning_rate,
                             weight_decay=1e-5)

    loss_list = []

    for epoch in range(num_epochs):
        for lmark, reye, leye, mouth, label in train_loader:

            if cuda:
                lmark = Variable(lmark).cuda().float()
                reye = Variable(reye).cuda().float()
                leye = Variable(leye).cuda().float()
                mouth = Variable(mouth).cuda().float()
                label = Variable(label).cuda().float()
            else:
                lmark = Variable(lmark).float()
                reye = Variable(reye).float()
                leye = Variable(leye).float()
                mouth = Variable(mouth).float()
                label = Variable(label).float()
            
            out, _ = model(lmark, reye, leye, mouth)
            out = out.view(out.size(0))

            loss = criterion(out, label)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            loss_list.append(loss.item())

        print('epoch [{}/{}], loss: {:.4f}'.format(
            epoch + 1,
            num_epochs,
            loss.item()))
    np.save('{}loss_list.npy'.format(class_dir), np.array(loss_list))
    torch.save(model.state_dict(), class_dir+'classifier.pth')



def test(in_dir, class_dir, feat_dir):

    if not os.path.exists(feat_dir):
        os.mkdir(feat_dir)
 
    cuda = torch.cuda.is_available()
    if cuda:
        print('cuda is available!')

    img_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_dataset = LD(in_dir, lmark_num = LMARK_NUM, color=1)
    test_loader = DataLoader(test_dataset, batch_size=3000)

    model = Encoder()
    if cuda:
        model.cuda()
    
    model.load_state_dict(torch.load(class_dir+'classifier.pth'))

    lmark, reye, leye, mouth, label = iter(test_loader).next()
    lmark = lmark.view(lmark.size(0), -1)
    label = label.numpy()
     
    with torch.no_grad():
        if cuda:
            lmark = Variable(lmark).cuda().float()
            reye = Variable(reye).cuda().float()
            leye = Variable(leye).cuda().float()
            mouth = Variable(mouth).cuda().float()
        else:
            lmark = Variable(lmark).float()
            reye = Variable(reye).float()
            leye = Variable(leye).float()
            mouth = Variable(mouth).float()
    
        out, feat = model(lmark, reye, leye, mouth)
        feat = feat.cpu().data.numpy()
        out = out.view(out.size(0)).cpu().data.numpy()
 
    f = open(class_dir + 'result.txt', 'w')
    for i in range(len(label)):
        f.write(str(label[i]) + ' ' + str(out[i]) + '\n')
        np.save(feat_dir+str(i).zfill(4)+'_'+str(int(label[i])), feat[i])
    f.close()
    evaluation(class_dir)



if __name__ == '__main__':
    mode = args.t
    in_dir = args.input
    finetuning = args.finetuning
    class_dir = args.classifier_dir
    feat_dir = args.feature

    print('mode       = ', mode)
    print('input      = ', in_dir)
    print('finetuning = ', finetuning)
    print('output     = ', class_dir)
    print('feature    = ', feat_dir)

    if mode == 'train':
        if finetuning:
            train(in_dir, class_dir, True)
        else:
            train(in_dir, class_dir, False)

    elif mode == 'test':
        test(in_dir, class_dir, feat_dir)
