from os import listdir
from os import mkdir
import cv2
import random
from copy import deepcopy
import numpy as np
from sklearn.decomposition import RandomizedPCA
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
class Data:
    def __init__(self, cut = 10):
        self.load(cut)

    def load(self,cut):
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        GaitDataSetPath = '/media/mithrandir/Mass Storage/BB/'
        people = listdir(GaitDataSetPath)
        self.gaits = {}
        self.GEI = {}
        self.raw={}
        count = 2
        for person in people[:]:
            self.gaits[person]=[]
            self.raw[person]=[]
            images = listdir(GaitDataSetPath+person+'/nm-01/072/')
            l=0
            for image in images:
                imagePath = GaitDataSetPath+person+'/nm-01/072/'+image
                img = cv2.imread(imagePath)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cntim,cnts,_ = cv2.findContours(deepcopy(img), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    if cv2.contourArea(c) < 1000:
			            continue
                    (x, y, w, h) = cv2.boundingRect(c)
                    # print (w, h)
                    dim = img[y:y+h, x:x+w]
                    dim=cv2.resize(dim, (64,128), interpolation = cv2.INTER_AREA)
                    dim = np.array(dim)
                    self.gaits[person].append(np.array(dim))
                    if count > 0:
                        if l%15==0:
                            # cv2.imwrite(person+image, dim)
                            l=0
                        l+=1
            count -= 1
            gaitimage = np.mean(self.gaits[person], axis=0)
            gaitimage.dtype = np.uint8
            cv2.imshow('az',gaitimage)

            des = hog.compute(gaitimage)
            self.GEI[person] = des.flatten()
            print person, "done", len(des)          # cv2.imwrite('Gaits/'+person+'-average.png',self.GEI[person])
        f=open('labels.txt','r')
        voo = f.read()
        voo = voo.split('\n')
        voo += ['1','1']
        self.labels =[]
        for i in range(len(self.GEI.keys())):
            self.labels.append(voo[int(self.GEI.keys()[i])-1])
        # avgimg = np.zeros((128,64),dtype=np.uint8)

        # avgimg = self.GEI[person]



class Model:
    def __init__(self,Data,n=1000):
        self.load(Data,n)
    def load(self,Data,n):
        data_set_biased = Data.GEI.values()
        data_set_labels_biased  = Data.labels
        data_set = []
        data_set_labels = []
        count = 0
        tup = []
        for i in range(len(data_set_biased)):
            if data_set_labels_biased[i]=='1':
                if count < 25:
                    tup.append((data_set_biased[i],'1'))
                    count += 1
            else:
                tup.append((data_set_biased[i],'0'))
        random.shuffle(tup)
        for i in range(len(tup)):
            data_set.append(tup[i][0])
            data_set_labels.append(tup[i][1])
        X_train, X_test, y_train, y_test = train_test_split(data_set, data_set_labels, test_size=0.25, random_state=datetime.now().second)
        # self.pca = RandomizedPCA(n_components=n,whiten=True).fit(X_train)
        # X_train_pca = self.pca.transform(X_train)
        # X_test_pca = self.pca.transform(X_test)
        X_train_pca = X_train
        X_test_pca = X_test
        self.scores = []
        self.clf = LinearSVC()
        self.clf.fit(X_train_pca,y_train)
        self.scores.append(self.clf.score(X_test_pca,y_test))

        self.clf = RandomForestClassifier()
        self.clf.fit(X_train_pca,y_train)
        self.scores.append(self.clf.score(X_test_pca,y_test))
        #
        # self.clf = MultinomialNB()
        # self.clf.fit(X_train_pca,y_train)
        # self.scores.append(self.clf.score(X_test_pca,y_test))

        self.clf = Perceptron()
        self.clf.fit(X_train_pca,y_train)
        self.scores.append(self.clf.score(X_test_pca,y_test))

        self.clf = AdaBoostClassifier()
        self.clf.fit(X_train_pca,y_train)
        self.scores.append(self.clf.score(X_test_pca,y_test))

        # for jo in range(len(self.scores)):
        #     print str(self.scores[jo])+'\t'
        print self.scores
        # print "Train Score ",s
        # print "Model succesfully Built."

d = Data()
for i in range(40):
    m=Model(d)
