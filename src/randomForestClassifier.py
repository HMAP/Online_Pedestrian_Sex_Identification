from sklearn.ensemble import RandomForestClassifier
from imageDescriptors import *
from os import listdir
files=sorted(listdir('data'))
X=[]
for file in files:
    d = Descriptor('data/'+file)
    X.append(d.des.flatten())
    del d

f = open('label.txt')
con = f.read()
labels = con.split('\n')
labels = labels[:-1]
for i in range(len(labels)):
    labels[i]=int(labels[i])
clf = RandomForestClassifier(n_estimators=10)
clf.fit(X[:800],labels[:800])
clf.score(X[800:900],labels[800:900])
