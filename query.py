# -*- coding: utf-8 -*-
from extract_cnn_vgg16_keras import VGGNet

import numpy as np
import h5py
import os
import  matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("-query", type=str, default='Caltech256/001_0001.jpg', help="Path to query which contains image to be queried")
    parser.add_argument("-index", type=str, default='featureCNN.h5' , help="Path to index")
    parser.add_argument("-result", type=str, default='Caltech256', help="Path for output retrieved images")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    # read in indexed images' feature vectors and corresponding image names
    opt = parse_opt()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    h5f = h5py.File(opt.index,'r')

    feats = h5f['feats'][:]
    # print(feats)

    imgNames = h5f['names'][:]
    print(imgNames)
    h5f.close()

    print("--------------------------------------------------")
    print("               searching starts")
    print("--------------------------------------------------")

    # read and show query image
    queryDir = opt.query
    queryImg = mpimg.imread(queryDir)
    plt.title("Query Image")
    plt.imshow(queryImg)
    plt.show()

    # init VGGNet16 model
    model = VGGNet()

    # extract query image's feature, compute simlarity score and sort
    queryVec = model.extract_feat(queryDir)
    scores = np.dot(queryVec, feats.T)
    rank_ID = np.argsort(scores)[::-1]
    rank_score = scores[rank_ID]
    #print rank_ID
    #print rank_score


    # number of top retrieved images to show
    maxres = 3
    print(enumerate(rank_ID[0:maxres]))
    imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:maxres])]
    print("top %d images in order are: " %maxres, imlist)

    # show top #maxres retrieved result one by one
    for i,im in enumerate(imlist):
        image = mpimg.imread(opt.result+"/"+str(im, 'utf-8'))
        plt.title("search output %d" %(i+1))
        plt.imshow(image)
        plt.show()
