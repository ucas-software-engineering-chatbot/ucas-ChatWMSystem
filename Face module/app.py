#coding: utf-8

from uuid import uuid4

from flask import Flask, render_template, request, send_from_directory
from flask_uploads import UploadSet, configure_uploads, IMAGES
import os
from keras.models import load_model
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from scipy.misc import imresize
import numpy as np
import cv2
import dlib
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
import tensorflow as tf
graph = tf.get_default_graph()
photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'
configure_uploads(app, photos)

# haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')

# 人脸识别模块
model_path = "/Users/suyuhui/Downloads/facial_beauty_prediction-master/web/data/mmod_human_face_detector.dat"
cnn_face_detector = dlib.cnn_face_detection_model_v1(model_path)

# 将人脸部分进行缩放，缩放到网络可以接受的大小
img_height, img_width, channels = 350, 350, 3

# 颜值识别模块
model = load_model("/Users/suyuhui/Downloads/mse-15-0.1046.h5")

# 检测人脸，并返回人脸区域的图片
def detect_face(filepath):
    im0 = cv2.imread(filepath)
    if im0.shape[0] > 1280:
        new_shape = (1280, im0.shape[1] * 1280 / im0.shape[0])
    elif im0.shape[1] > 1280:
        new_shape = (im0.shape[0] * 1280 / im0.shape[1], 1280)
    elif im0.shape[0] < 640 or im0.shape[1] < 640:
        new_shape = (im0.shape[0] * 2, im0.shape[1] * 2)
    else:
        new_shape = im0.shape[0:2]
    im = cv2.resize(im0, (int(new_shape[1]), int(new_shape[0])))
    dets = cnn_face_detector(im, 0)
    if len(dets) != 1:
        return np.zeros((1,1))
    rect = dets[0].rect
    height = rect.bottom() - rect.top()
    boundary = round(height*0.8)
    # head_top = round(rect.top() - boundary)
    if rect.top() - boundary < 0:
        # head_top = 0
        boundary = rect.top()
    # head_bottom = round(rect.bottom() + boundary)
    if rect.bottom() + boundary >= int(new_shape[0]):
        # head_bottom = int(new_shape[0])
        boundary = int(new_shape[0]) - rect.bottom()
    # head_left = round(rect.left() - boundary)
    if rect.left() - boundary < 0:
        # head_left = 0
        boundary = rect.left()
    # head_right = round(rect.right() + boundary)
    if rect.right() + boundary >= int(new_shape[1]):
        # head_right = int(new_shape[1])
        boundary = int(new_shape[1]) - rect.right()
    cropped_image = im[rect.top()-boundary:rect.bottom()+boundary,
                    rect.left()-boundary:rect.right()+boundary, :]
    # img = load_img(filepath)
    # img = imresize(img, size=(img_height, img_width))
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5)
    # if len(faces) == 0:
    #     return np.zeros((1,1))
    # x, y, w, h = faces[0]
    # cropped_image = img[y:y + h, x:x + w, :]
    resized_image = cv2.resize(cropped_image, (img_height, img_width))
    # resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    filename = filepath.split('/')[-1]
    cv2.imwrite("uploads/cropped_{}".format(filename), resized_image)
    return resized_image

# 返回颜值分数
# filepath: 图片路径
def get_score(filepath):
    test_x = detect_face(filepath)
    if not test_x.any():
        return None
    test_x = test_x / 255.
    test_x = test_x.reshape((1,) + test_x.shape)
    with graph.as_default():
        predicted = model.predict(test_x)
    base = predicted[0][0]
    if predicted[0][0]>=3.5:
        base = ((base-3.5)/1.5)*10
        score = 90+base
    elif base<3.5 and base>=2.5:
        base = ((3.5-base) / 1) * 30
        score = 90-base
    else:
        base = ((2.5 - base) / 2.5) * 60
        score = 60 - base

    score = round(score)
    # score = round((predicted[0][0]-5.0)*60+40)
    # if score>=100:
    #     score = 95.10
    print(str(score)+", "+str(round(predicted[0][0],2)))
    return score

# 默认的主界面
@app.route("/")
def index():
    return render_template("upload.html", image_name="demo.jpeg", score=4.04)

# 上传图片的路由
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files.get('photo'), name="{}.".format(str(uuid4())))
        score = get_score("uploads/{}".format(filename))
        if not score:
            return render_template("error.html")
        else:
            return render_template("upload.html", image_name="{}".format(filename), score=score)

# 应该没啥用
@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory("uploads", filename)

# flask入口
if __name__ == '__main__':
    # WSGIServer(('127.0.0.1', 3000), app).serve_forever()
    app.run(host="127.0.0.1", port=3000, debug=True)
