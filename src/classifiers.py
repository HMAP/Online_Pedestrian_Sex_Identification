from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from imageDescriptors import *
from os import listdir
import cv2
files=sorted(listdir('data'))
X=[]
hog = cv2.HOGDescriptor()
for file in files:
    #d = Descriptor('data/'+file)
    im = cv2.imread('data/'+file)
    #img_data = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    img_data =im
    des = hog.compute(img_data)
    X.append(des.flatten())
    #del d
print "Image Descritors Have been generated"
f = open('label.txt')
con = f.read()
labels = con.split('\n')
labels = labels[:-1]
for i in range(len(labels)):
    labels[i]=int(labels[i])
rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(X[:900],labels[:900])
print "RandomForestClassifier",rfc.score(X[900:],labels[900:])
gnb = GaussianNB()
gnb.fit(X[:900],labels[:900])
print "GaussianNB",gnb.score(X[900:],labels[900:])
m = Perceptron()
m.fit(X[:900],labels[:900])
print "MultinomialNB", m.score(X[900:],labels[900:])
p = Perceptron()
p.fit(X[:900],labels[:900])
print "Perceptron", p.score(X[900:],labels[900:])
a = AdaBoostClassifier()
a.fit(X[:900],labels[:900])
print "Adaboost", a.score(X[900:],labels[900:])
