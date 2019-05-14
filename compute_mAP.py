from extract_cnn_vgg16_keras import VGGNet

import numpy as np
import h5py
import os
import argparse
import time

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("-index", type=str, default='./featureCNN_map.h5' , help="Path to index")
    parser.add_argument("-bathPath", type=str, default='./database/' , help="bath_path")
    parser.add_argument("-queryFile", type=str, default='./queryImgs.txt' , help="queryFile")
    parser.add_argument("-classesFile", type=str, default='./databaseClasses.txt' , help="classesFile")
    parser.add_argument("-topk", type=int, default=20, help='topK')
    parser.add_argument("-gpu", type=str, default="5", help='which gpu to use| set "" if use cpu')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    opt = parse_opt()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    print('Loading index...')
    # init VGGNet16 model
    model = VGGNet()
    start_time = time.time()
    h5f = h5py.File(opt.index, 'r')
    feats = h5f['feats'][:]
    imgNames = h5f['names'][:]
    h5f.close()
    print("finished loding index", time.time() - start_time)

    basepath = opt.bathPath
    queryFile = opt.queryFile
    classesFile = opt.classesFile
    N=opt.topk

    queryImgs = []
    classesAndNum =[]
    with open(queryFile, 'r') as f:
        for line in f:
            line = line.strip('\n')
            queryImgs.append(line)

    with open(classesFile, 'r') as f:
        for line in f:
            line = line.strip('\n')
            classesAndNum.append(line)
    # 001_001.ak47 98  002_002.american-flag 97

    classes = []  #['001','002',...,'257']
    for i in range(0,len(classesAndNum)):
        classes.append(classesAndNum[i][0:3])

    querysNum=len(queryImgs) #15247

    ap=np.zeros(querysNum)

    total_start = time.time()
    for i in range(0, querysNum):
        start = time.time()
        queryName = basepath+queryImgs[i]
        # print(queryName)
        queryClass = queryImgs[i][0:3]
        # print(queryClass)
        # extract query image's feature, compute simlarity score and sort
        queryFeat = model.extract_feat(queryName)
        # print(classesAndNum[0].size)
        queryClassNum = classesAndNum[classes.index(queryClass)].split(' ')[1]

        scores = np.dot(queryFeat, feats.T)
        rank_ID = np.argsort(scores)[::-1]
        rank_score = scores[rank_ID]

        # number of top retrieved images to showN
        imlist = [imgNames[index] for i, index in enumerate(rank_ID[0:N])]
        # print(imlist)

        similarTerm = 0
        precision = np.zeros(N)

        for k in range(0,N):
            topkClass = imlist[k][0:3].decode('utf-8')
            if queryClass==topkClass:
                similarTerm = similarTerm +1
                precision[k] = similarTerm/(k+1)

        # retrievalSamples = np.sum(list(map(lambda x: x >= 0, precision)))
        # print(retrievalSamples)
        ap[i] = np.sum(precision)
        ap[i] = np.true_divide(ap[i], int(N))
        cost_time = time.time() - start
        print('queryName: {} ap: {} Time:{:.3f} (s)'.format(queryImgs[i],ap[i],cost_time))

    mAP = np.sum(ap)/ int(querysNum)
    total_cost = time.time()-total_start
    print('mAP:{} of {} query, total cost: {} '.format(mAP, querysNum, total_cost))