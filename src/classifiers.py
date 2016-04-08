from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from os import listdir
import cv2
class Data:
    def __init__(self):
        files=sorted(listdir('data'))
        X=[]
        hog = cv2.HOGDescriptor()
        for file in files:
            im = cv2.imread('data/'+file)
            img_data = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            des = hog.compute(img_data)
            X.append(des.flatten())
        print "Image Descritors Have been generated"
        f = open('label.txt')
        con = f.read()
        labels = con.split('\n')
        labels = labels[:-1]
        for i in range(len(labels)):
            labels[i]=int(labels[i])
        self.X = X
        self.y = labels

class Model:
    def __init__(self,Data):
        X_train, X_test, y_train, y_test = train_test_split(Data.X, Data.y, test_size=0.1, random_state=42)
        self.clf = LinearSVC()
        self.clf.fit(X_train,y_train)
        print self.clf.score(X_test,y_test)
