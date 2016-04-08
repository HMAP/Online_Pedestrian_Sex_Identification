import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from data import *
import random
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

class rC:
	def __init__(self,Data):
		tups = []
		for i in Data.males:
			tups.append([i,1])
		for i in Data.females:
			tups.append([i,0])
		X=[]
		y=[]
		random.shuffle(tups)
		for s,l in tups:
			X.append(s)
			y.append(l)
		X = np.array(X).astype(np.uint8)
		y = np_utils.to_categorical(np.array(y),2)
		X = X.reshape(X.shape[0],1,32,32)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state = 42)
		print len(X_train),len(y_train)
		print X.shape,y.shape

		self.model = Sequential()

		self.model.add(Convolution2D(64,4,4, border_mode='same',input_shape=(1,32,32)))
		self.model.add(Convolution2D(32,8,8, border_mode='same'))
		self.model.add(Convolution2D(16,16,16, border_mode='same'))
		self.model.add(Flatten())
		self.model.add(Dense(32))
		self.model.add(Dense(2))
		self.model.add(Activation('softmax'))

		# let's train the model using SGD + momentum (how original).
		sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		self.model.compile(loss='categorical_crossentropy', optimizer=sgd)

		self.model.fit(X_train, y_train, batch_size=16,nb_epoch=5, show_accuracy=True,validation_split=0.1, shuffle=True)
		print self.model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)

r=rC(d)
