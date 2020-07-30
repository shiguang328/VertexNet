#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import os
import torch
from torch.utils import data
from PIL import Image
from torchvision import transforms


def readfile(root, label_file):
    f = open(label_file, 'r', encoding='utf-8')
    ims, labels = [], []
    for line in f.readlines():
        line = line.strip('\n').strip('\r')
        if line == '':
            continue
        jpg_index = line.find('jpg')
        img_name = line[0:jpg_index + 3]
        label_str = line[jpg_index + 4:]
        img_path = os.path.join(root, img_name)
        ims.append(img_path)
        labels.append(label_str)
    dic = {}
    for i, j in enumerate(ims):
        dic[ims[i]] = labels[i]
    return dic


def resize(img, label):
    if img.size[0] != 224 or img.size[1] != 244:
        x_ratio = int(224 / img.size[1])
        y_ratio = int(224 / img.size[0])

        lab = []
        for i in range(4):
            lab.append(x_ratio * label[i * 2])
            lab.append(y_ratio * label[i * 2 + 1])
        return lab
    else:
        return label


class ResizeNormalize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class MyDataSet(data.Dataset):
    def __init__(self, imgroot, label):
        self.img_dict = readfile(imgroot, label)
        self.img_root = imgroot
        self.img_name = [filename for filename, _ in self.img_dict.items()]

    def __getitem__(self, index):
        img_path = self.img_name[index]
        keys = self.img_dict.get(self.img_name[index])
        image = Image.open(img_path).convert('L')
        keys = resize(image, keys)
        transform = ResizeNormalize((224, 224))
        image = transform(image)
        return image, keys

    def __len__(self):
        return len(self.img_name)


def load_data(img_dir, label_file, batch_size):

    torch_dataset = MyDataSet(img_dir, label_file)
    loader = data.DataLoader(dataset=torch_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=1,)

    return loader


def resize_t(img, label):
    if img.shape[0] != 224 or img.shape[1] != 244:
        image = cv2.resize(img, (224, 224))
        x_ratio = float(224 / img.shape[1])
        y_ratio = float(224 / img.shape[0])

        lab = []
        for i in range(4):
            lab.append(float(x_ratio * label[i * 2]/224))
            lab.append(float(y_ratio * label[i * 2 + 1]/224))

        return image, lab
    else:
        return img, label


def load_data_t(img_dir, label_file, batch_size):

    x_lst, y_lst = [], []
    f = open(label_file, 'r', encoding='utf-8')
    for line in f.readlines():
        line = line.strip('\n').strip('\r')
        if line == '':
            continue
        jpg_index = line.find('jpg')
        img_name = line[0:jpg_index+3]
        label_str = line[jpg_index+4:]
        im = cv2.imread(img_dir + img_name)
        label = [int(item) for item in label_str.replace(',', ' ').split(' ')]
        ims, labels = resize_t(im, label)
        x_lst.append(ims)
        y_lst.append(labels)

    torch_dataset = data.TensorDataset(torch.Tensor(x_lst), torch.Tensor(y_lst))
    loader = data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )

    return loader
