#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import torch
import rgcn.net as net
import numpy as np


model_path = '../data/model/netRGCN_1.pth'

model = net.load()
model.load_state_dict(torch.load(model_path, map_location='cpu'))


# rgcn orignal output
def rgcn(image):
    x = cv2.resize(image, (224, 224))
    x = torch.from_numpy(x)
    x = x.unsqueeze(0)
    x = x.permute(0, 3, 1, 2)
    x = x.float()
    y_pred = model(x)
    return y_pred.detach().numpy()[0]


#posi trans & warp trans
def warp(image):
    img = cv2.imread(image)
    result = rgcn(img)
    if result is not None and len(result) == 8:
        size_w, size_h = 224, 224
        h, w = img.shape[0], img.shape[0]
        new_pos_lst = []
        for i in range(4):
            w_1 = max(result[i*2], 0)
            h_1 = max(result[i*2+1], 0)
            new_pos_lst.append(int(w * w_1))
            new_pos_lst.append(int(h * h_1))

        pts1 = np.float32([[new_pos_lst[0], new_pos_lst[1]], [new_pos_lst[2], new_pos_lst[3]],
                           [new_pos_lst[4], new_pos_lst[5]], [new_pos_lst[6], new_pos_lst[7]]])
        pts2 = np.float32([[0, 0], [800, 0], [0, 500], [800, 500]])

        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, M, (800, 500))
        return dst


if __name__ == '__main__':
    path = r'../data/images/truth_10050.jpg'
    im = warp(path)
    cv2.imwrite('../data/test.jpg', im)
    cv2.imshow('OK', im)
    cv2.waitKey(0)


