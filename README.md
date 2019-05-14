# Content Based Image Retrieval System with Convolutional Neural Networks
It is the code for a CBIR system with CNN model. We provide a demo of a web application based on Flask, which realizes the online CBIR system for receiving the uploaded images and showing the most similar top 15 pictures. 

## Requirements
- Python 3.6
- Keras 2.0.5
- TensorFlow (1.8.0)
- cuda & cudnn (optional, only using the feature extraction)

You can install all python requirements with:
```
pip install -r requirements.txt
```
## Dataset
Caltech-256 (http://www.vision.caltech.edu/Image_Datasets/Caltech256/)

## Usage
### Feature extraction and build index
```
python index.py
```
### Query test on local
```
python query.py
```
### Calcuate the mAP
```
python compute_mAP.py
```
### Run web-server
```
python server.py
```

## Demo (Please wait serval seconds for downloading)
![image](https://github.com/ruizewang/CBIR-CNN/blob/master/demo.gif )   
