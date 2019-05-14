import sys
sys.path.append("..")
from extract_cnn_vgg16_keras import VGGNet
import numpy as np

import h5py
import os
import  matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import time

import flask
from flask import *
from flask.json import jsonify
# import cv2
from werkzeug.utils import secure_filename

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8');
        return json.JSONEncoder.default(self, obj)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("-query", type=str, default='../Caltech256/001_0001.jpg', help="Path to query which contains image to be queried")
    parser.add_argument("-index", type=str, default='../featureCNN.h5' , help="Path to index")
    parser.add_argument("-result", type=str, default='../Caltech256', help="Path for output retrieved images")
    args = parser.parse_args()

    return args

app = flask.Flask("image-retrieval")

opt = parse_opt()
os.environ["CUDA_VISIBLE_DEVICES"] = ""

print('Loading index...')
# init VGGNet16 model
model = VGGNet()
start_time = time.time()
h5f = h5py.File(opt.index, 'r')

feats = h5f['feats'][:]
imgNames = h5f['names'][:]

h5f.close()
print("finished loding index", time.time() - start_time)

# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['POST', 'GET'])  # 添加路由
def upload():
    if request.method == 'POST':
        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        user_input = request.form.get("name")

        basepath = os.path.dirname(__file__)  # 当前文件所在路径

        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        # upload_path = os.path.join(basepath, 'static/images','test.jpg')  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)

        # 使用Opencv转换一下图片格式和名称
        # img = cv2.imread(upload_path)
        # cv2.imwrite(os.path.join(basepath, 'app/static/images', 'test.jpg'), img)
        img_url=os.path.join(basepath, 'static/images', 'test.jpg')
        # cv2.imwrite(img_url, img)
        start_time = time.time()
        im = get_preditction(upload_path)
        print("Prediction cost", time.time() - start_time)
        # print(im)

        return render_template('result.html',images=json.dumps(im,cls=MyEncoder),upload_image=secure_filename(f.filename))

    return render_template('upload.html')

def get_preditction(img_url):
    print("--------------------------------------------------")
    print("               searching starts")
    print("--------------------------------------------------")

    # read and show query image
    basepath = os.path.dirname(__file__)  # 当前文件所在路径
    queryDir = img_url
    queryImg = mpimg.imread(queryDir)

    # extract query image's feature, compute simlarity score and sort
    queryVec = model.extract_feat(queryDir)
    scores = np.dot(queryVec, feats.T)
    rank_ID = np.argsort(scores)[::-1]
    rank_score = scores[rank_ID]

    # number of top retrieved images to show
    maxres = 15
    print(enumerate(rank_ID[0:maxres]))
    imlist = [imgNames[index] for i, index in enumerate(rank_ID[0:maxres])]
    print(imlist[0])
    print("top %d images in order are: " % maxres, imlist)
    return imlist

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=7777, debug=True)
