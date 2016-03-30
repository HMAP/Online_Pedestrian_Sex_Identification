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
class Data:
    def __init__(self):
        self.load()

    def load(self):
        self.males = []
        for male in listdir('data/faces94/male'):
            for file in listdir('data/faces94/male/'+male)[:int(sys.argv[1])]:
                im = Image.open('data/faces94/male/'+male+'/'+file,'r')
                im = im.convert('L')
                self.males.append(np.asarray(im))

        self.females = []
        for female in listdir('data/faces94/female'):
            for file in listdir('data/faces94/female/'+female)[:int(sys.argv[1])]:
                im = Image.open('data/faces94/female/'+female+'/'+file,'r')
                im= im.convert('L')
                self.females.append(np.asarray(im))

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
class eF:
    def __init__(self,Data):
        self.load(Data)

    def load(self,Data):
        data_set = []
        data_set_labels = []
        for person in Data.males:
            data_set.append(person.flatten())
            data_set_labels.append(1)
        for person in Data.females:
            data_set.append(person.flatten())
            data_set_labels.append(-1)
        h,w=person.shape
        X_train, X_test, y_train, y_test = train_test_split(data_set, data_set_labels, test_size=0.25, random_state=datetime.now().second)

        n_components = 15

        pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        eigenfaces = pca.components_.reshape((n_components, h, w))
        eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
        #plot_gallery(eigenfaces, eigenface_titles, h, w)
        clf = LinearSVC()
        clf.fit(X_train_pca,y_train)
        s=clf.score(X_test_pca,y_test)
        print s

a=Data()
b=eF(a)
