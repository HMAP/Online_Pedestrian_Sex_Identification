from os import listdir
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import LinearSVC
from datetime import datetime
import sys
import cv2
from array import array

class Data:
    def __init__(self):
        self.load()

    def load(self):

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        self.males = []
        for file in listdir('data/male'):
            gray = Image.open('data/male/'+file,'r')
            gray=gray.convert('L')
            gray=np.asarray(gray)
            try:
                (x,y,w,h) = face_cascade.detectMultiScale(gray, 1.3, 5)[0]
                face = gray[y:y+h, x:x+w]
                dim=cv2.resize(face, (100, 100), interpolation = cv2.INTER_AREA)
                self.males.append(np.array(dim))
                cv2.imwrite('data/cropped/male/'+file+'.jpg', dim)
            except:
                pass
                print 'male',file

        self.females = []
        for file in listdir('data/female'):
            gray = Image.open('data/female/'+file,'r')
            gray=gray.convert('L')
            gray=np.asarray(gray)
            try:
                (x,y,w,h) = face_cascade.detectMultiScale(gray, 1.3, 5)[0]
                face = gray[y:y+h, x:x+w]
                dim=cv2.resize(face, (100, 100), interpolation = cv2.INTER_AREA)
                self.females.append(np.array(dim))
                cv2.imwrite('data/cropped/female/'+file+'.jpg', dim)
            except:
                pass
                print 'female',file

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.show()

class model:
    def __init__(self,Data):
        self.load(Data)

    def load(self,Data):
        data_set = []
        data_set_labels = []
        for person in Data.males:
            data_set.append(person.flatten())
            data_set_labels.append('male')
        for person in Data.females:
            data_set.append(person.flatten())
            data_set_labels.append('female')
        h,w=person.shape
        X_train, X_test, y_train, y_test = train_test_split(data_set, data_set_labels, test_size=0.1, random_state=datetime.now().second)

        n_components = 15

        self.pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
        X_train_pca = self.pca.transform(X_train)
        X_test_pca = self.pca.transform(X_test)
        eigenfaces = self.pca.components_.reshape((n_components, h, w))
        eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
        #plot_gallery(eigenfaces, eigenface_titles, h, w)
        self.clf = LinearSVC()
        self.clf.fit(X_train_pca,y_train)
        print X_test_pca[0].shape
        s=self.clf.score(X_test_pca,y_test)
        print s

class video:
    def __init__(self,model):
        self.load(model)

    def load(self,model):
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        while True:
            flag, frame = cap.read()
            if flag:
                # The frame is ready and already captured
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in faces:
                    face = gray[y:y+h, x:x+w]
                    dim=cv2.resize(face, (100, 100), interpolation = cv2.INTER_AREA)
                    dim = np.array(dim)
                    dim=dim.flatten()
                    print dim
                    transform = model.pca.transform([dim])[0]
                    print transform.shape
                    prediction = model.clf.predict([transform])[0]
                    # print prediction
                    if prediction == 'male':
                        colo = (255,0,0)
                        print 'male'
                    else:
                        colo = (0,0,255)
                        print "female"
                    cv2.rectangle(frame,(x,y),(x+w,y+h),colo,2)
                cv2.imshow('video', frame)
                pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                print str(pos_frame)+" frames"
            else:
                # The next frame is not ready, so we try to read it again
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
                print "frame is not ready"
                # It is better to wait for a while for the next frame to be ready
                cv2.waitKey(1000)

            if cv2.waitKey(10) == 27:
                break
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                # If the number of captured frames is equal to the total number of frames,
                # we stop
                break


a=Data()
b=model(a)
c=video(b)
