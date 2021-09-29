import os
import torch
import logging
import cv2
from PIL import Image
import imageio
import numpy as np
import torch.utils.data as data
from os.path import join, exists
import math
import random
import sys
import json
import random
from subnet.basics import *
from subnet.ms_ssim_torch import ms_ssim
from augmentation import random_flip, random_crop_and_pad_image_and_labels

class DataSet(data.Dataset):
    def __init__(self, path="./data/vimeo_septuplet/test.txt", im_height=256, im_width=256):
        self.image_input_list, self.image_ref_list = self.get_vimeo(filefolderlist=path)
        self.im_height = im_height
        self.im_width = im_width

        print("dataset find image: ", len(self.image_input_list))

    def get_vimeo(self, rootdir="./vimeo_septuplet/sequences/", filefolderlist="./data/vimeo_septuplet/test.txt"):
        with open(filefolderlist) as f:
            data = f.readlines()
            
        fns_train_input = []
        fns_train_ref = []

        for n, line in enumerate(data, 1):
            y = os.path.join(rootdir, line.rstrip())
            fns_train_input += [y]
            refnumber = int(y[-5:-4]) - 2
            refname = y[0:-5] + str(refnumber) + '.png'
            fns_train_ref += [refname]

        return fns_train_input, fns_train_ref

    def __len__(self):
        return len(self.image_input_list)

    def __getitem__(self, index):
        input_image = imageio.imread(self.image_input_list[index])
        ref_image = imageio.imread(self.image_ref_list[index])

        input_image = input_image.astype(np.float32) / 255.0
        ref_image = ref_image.astype(np.float32) / 255.0

        input_image = input_image.transpose(2, 0, 1)
        ref_image = ref_image.transpose(2, 0, 1)
        
        input_image = torch.from_numpy(input_image).float()
        ref_image = torch.from_numpy(ref_image).float()

        input_image, ref_image = random_crop_and_pad_image_and_labels(input_image, ref_image, [self.im_height, self.im_width])
        input_image, ref_image = random_flip(input_image, ref_image)

        return input_image, ref_image

class HEVCDataSet(data.Dataset):
    def __init__(self, root="./data/hevctest/images", refdir='./data/hevctest/videos_crop/', crf='H265QP27', testfull=False):

        self.ref = []
        self.refbpp = []
        self.input = []
        self.hevcclass = ['BasketballDrive', 'BQTerrace', 'Cactus', 'Kimono1', 'ParkScene']
        AllIbpp = self.getbpp(crf)
        ii = 0
        for seq in self.hevcclass:
            seqIbpp = AllIbpp[ii]
            imlist = os.listdir(os.path.join(refdir, seq, crf))
            cnt = 0
            for im in imlist:
                if im[-4:] == '.png':
                    cnt += 1
            if testfull:
                framerange = cnt // 10
            else:
                framerange = 1
            for i in range(framerange):
                refpath = os.path.join(refdir, seq, crf, 'im' + str(i * 10 + 1).zfill(4) + '.png')
                inputpath = []
                for j in range(10):
                    inputpath.append(os.path.join(root, seq, 'im' + str(i * 10 + j + 1).zfill(3) + '.png'))
                self.ref.append(refpath)
                self.refbpp.append(seqIbpp)
                self.input.append(inputpath)
            ii += 1

    def getbpp(self, crf):
        Ibpp = None
        if crf == 'H265QP22':
            print('use H265QP22')
            Ibpp = []  # you need to fill bpps after generating QP=22
        elif crf  == 'H265QP27':
            print('use H265QP27')
            Ibpp = [1.2450358072916667, 0.43555094401041666, 0.9358439127604167, 0.34054361979166664, 0.8700154622395834]  # you need to fill bpps after generating QP=27
        elif crf  == 'H265QP32':
            print('use H265QP32')
            Ibpp = []  # you need to fill bpps after generating QP=32
        elif crf  == 'H265QP37':
            print('use H265L29')
            Ibpp = []  # you need to fill bpps after generating QP=37
        else:
            print('cannot find ref : ', crf)
            exit()
        if len(Ibpp) == 0:
            print('You need to generate I frames and fill the bpps above!')
            exit()
        return Ibpp

    def __len__(self):
        return len(self.ref)

    def __getitem__(self, index):
        ref_image = imageio.imread(self.ref[index]).transpose(2, 0, 1).astype(np.float32) / 255.0
        h = (ref_image.shape[1] // 64) * 64
        w = (ref_image.shape[2] // 64) * 64
        ref_image = np.array(ref_image[:, :h, :w])
        input_images = []
        refpsnr = None
        refmsssim = None
        for filename in self.input[index]:
            input_image = (imageio.imread(filename).transpose(2, 0, 1)[:, :h, :w]).astype(np.float32) / 255.0
            if refpsnr is None:
                refpsnr = CalcuPSNR(input_image, ref_image)
                refmsssim = ms_ssim(torch.from_numpy(input_image[np.newaxis, :]),
                                    torch.from_numpy(ref_image[np.newaxis, :]), data_range=1.0).numpy()
            else:
                input_images.append(input_image[:, :h, :w])

        input_images = np.array(input_images)
        return input_images, ref_image, self.refbpp[index], refpsnr, refmsssim
