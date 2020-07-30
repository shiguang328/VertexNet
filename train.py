#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.optim as optim
import rgcn.data as data
import rgcn.net as net
from rgcn.loss import L2_Loss

# train parameters
parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', required=False, default='../data/images/', help='path to images')
parser.add_argument('--train', required=False, default='../data/label/train.txt', help='path to train')
parser.add_argument('--test', required=False, default='../data/label/val.txt', help='path to test')
parser.add_argument('--model_dir', required=False, default='../data/model/', help='path to model')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--cnt_epochs', type=int, default=100, help='iter numbers')
parser.add_argument('--cnt_save', type=int, default=1, help='Interval to be save model')
parser.add_argument('--cnt_val', type=int, default=1, help='Interval to be val model')
opt = parser.parse_args()
# print(opt)

if opt.model_dir is None:
    opt.model_dir = r'../data/model/'
# os.system('mkdir {0}'.format(opt.model_dir))

# load data: images and label
train_loader = data.load_data_t(opt.img_dir, opt.train, opt.batch_size)
test_loader = data.load_data_t(opt.img_dir, opt.test, opt.batch_size)
#load model
model = net.load()
#creat loss
criterion = L2_Loss()
# GPU or CPU
if torch.cuda.is_available():
    model.cuda()
    criterion.cuda()

#creat optimizer
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))


def calculate(data):
    # read data
    cal_iter = iter(data)
    x, y_ = cal_iter.next()
    x = x.permute(0, 3, 1, 2)
    # input
    if torch.cuda.is_available():
        x = x.cuda()
    y = model(x)
    # calculate loss
    if torch.cuda.is_available():
        y_ = y_.cuda()
    loss = criterion(y, y_)

    return loss


if __name__ == '__main__':
    #train
    cnt_batch = len(train_loader) + 1
    for epoch in range(opt.cnt_epochs):
        for i in range(cnt_batch):
            loss = calculate(train_loader)
            #clear grad
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Train Epoch: [%d/%d], Batch: [%d/%d], Loss: %f' % (epoch+1, opt.cnt_epochs, i+1, cnt_batch,loss))

        #val
        if (epoch + 1)%opt.cnt_val == 0:
            num_val = len(test_loader) + 1
            sum_loss = 0
            for i in range(num_val):
                loss = calculate(test_loader)
                sum_loss += float(loss)
            print('Val Epoch: [%d/%d],  Loss: %f' % (epoch + 1, opt.cnt_epochs, sum_loss/float(opt.cnt_val)))

        #save model
        if (epoch + 1)%opt.cnt_save == 0:
            torch.save(model.state_dict(), '{0}/netRGCN_{1}.pth'.format(opt.model_dir, epoch+1))
            print('Save model {0}/netRGCN_{1}.pth'.format(opt.model_dir, epoch+1))

