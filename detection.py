from model import create_model
from align import AlignDlib

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os


class Detector:
    def __init__(self, metadata):
        self.nn4_small2_pretrained = create_model()
        self.nn4_small2.load_weights('weights\nn4.small2.v1.h5')
        self.alignment = AlignDlib('models/landmarks.dat')
        self.svm_model = LinearSVC()

        self.recognizer = np.zeros((metadata.shape[0]), 128)

        embed_vec(metadata, self.recognizer)
        
        train_recog(metadata, self.recognizer, self.svm_model)


    def recog_face(self, path):
        img_rgb = load_image(path)
        bb = alignment.getLargestFaceBoundingBox(img_rgb)
        return bb


    def embed_vec(self, metadata, recog):
        for i,m in enumerate(metadata):
            img = load_image(m.image_path())
            img = align_image(img)
            img = (img/255.).astype(np.float32)
            recog[i] = self.nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]


    def load_image(self, path):
        img = cv.imread(path,1)
        return img[...,::-1]


    def align_image(self, img):
        return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), 
        landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

    def train_recog(self, metadata, recog, model):
        targets = np.array([m.name for m in metadata])

        encoder = LabelEncoder()
        encoder.fit(targets)

        # Numerical encoding of identities
        y = encoder.transform(targets)

        train_idx = np.arange(metadata.shape[0]) % 2 != 0
        test_idx = np.arange(metadata.shape[0]) % 2 == 0

        # 50 train examples of 10 identities (5 examples each)
        X_train = recog[train_idx]
        # 50 test examples of 10 identities (5 examples each)
        X_test = recog[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        svc = LinearSVC()

        svc.fit(X_train, y_train)

        acc_svc = accuracy_score(y_test, svc.predict(X_test))


class IdentityMetaData:
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file) 