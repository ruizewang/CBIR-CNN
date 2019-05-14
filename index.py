# -*- coding: utf-8 -*-
import os
import h5py
import numpy as np
import argparse
import time

from extract_cnn_vgg16_keras import VGGNet


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("-database", type=str, default='./database', help="Path to database which contains images to be indexed")
    # parser.add_argument("-database", type=str, default='./web-server/static/Caltech256', help="Path to database which contains images to be indexed")
    parser.add_argument("-index", type=str, default='featureCNN_map.h5', help="Name of index file")

    args = parser.parse_args()

    return args

'''
 Returns a list of filenames for all jpg images in a directory. 
'''
def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]


'''
 Extract features and index the images
'''
if __name__ == "__main__":
    opt = parse_opt()
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    db = opt.database
    img_list = get_imlist(db)

    print("--------------------------------------------------")
    print("         feature extraction starts")
    print("--------------------------------------------------")
    start = time.time()

    feats = []
    names = []

    model = VGGNet()
    for i, img_path in enumerate(img_list):
        norm_feat = model.extract_feat(img_path)
        # 512维向量(512,)
        img_name = os.path.split(img_path)[1]
        feats.append(norm_feat)
        names.append(img_name)
        print("extracting feature from image No. %d , %d images in total" %((i+1), len(img_list)))

    feats = np.array(feats)

    # directory for storing extracted features
    output = opt.index
    
    print("--------------------------------------------------")
    print("      writing feature extraction results ...")
    print("--------------------------------------------------")


    h5f = h5py.File(output, 'w')
    h5f.create_dataset('feats', data = feats)
    h5f.create_dataset('names', data = np.string_(names))
    h5f.close()
    end = time.time()
    print("Cost time: ",end-start," (s)")

