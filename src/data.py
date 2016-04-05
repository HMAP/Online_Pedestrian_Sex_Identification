from os import listdir
from os import mkdir
import cv2
from copy import deepcopy
import numpy as np
from sklearn.decomposition import RandomizedPCA
class Data:
    def __init__(self, cut = 10):
        self.load(cut)

    def load(self,cut):
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        GaitDataSetPath = '..\\GaitDatasetA-silh\\silhouettes\\'
        people = listdir(GaitDataSetPath)
        self.gaits = {}
        self.GEI = {}
        self.raw={}
        for person in people[:]:
            self.gaits[person]=[]
            self.raw[person]=[]
            images = listdir(GaitDataSetPath+person+'\\00_1')
            for image in images:
                # f = open(GaitDataSetPath+person+'\\00_1\\'+image,'r')
                imagePath = GaitDataSetPath+person+'\\00_1\\'+image
                img = cv2.imread(imagePath)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # img = cv2.GaussianBlur(img, (21, 21), 0)
                # img = cv2.dilate(img, None, iterations=2)
                #print 'yeah'
                cntim,cnts,_ = cv2.findContours(deepcopy(img), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                #print len(cnts),person
                # print cnts.shape
                # print cnts
                for c in cnts:
                    if cv2.contourArea(c) < 1000:
			            continue
                    (x, y, w, h) = cv2.boundingRect(c)
                    print (w, h)
                    dim = img[y:y+h, x:x+w]
                    dim=cv2.resize(dim, (64,128), interpolation = cv2.INTER_AREA)
                    dim = np.array(dim)
                    self.gaits[person].append(np.array(dim))

                print len(self.gaits[person])
                # cv2.imwrite(person+image+'.png', dim)
        # avgimg = np.zeros((100,100),dtype=np.uint8)
        self.GEI[person] = np.mean(self.gaits[person], axis=0)
        # cv2.imwrite(person+'-average.png',avg)

class Model:
    def __init__(self,Data,n):
        self.load(Data,n)
    def load(self,Data):
        data_set = Data.GEI.values()
        data_set_labels  = Data.GEI.keys()
        X_train, X_test, y_train, y_test = train_test_split(data_set, data_set_labels, test_size=0.1, random_state=datetime.now().second)
        self.pca = RandomizedPCA(n_components=n,whiten=True).fit(X_train)
        X_train_pca = self.pca.transform(X_train)
        X_test_pca = self.pca.transform(X_test)
        self.clf = LinearSVC()
        self.clf.fit(X_train_pca,y_train)
        s=self.clf.score(X_test_pca,y_test)
        print "Train Score ",s
        print "Model succesfully Built."
