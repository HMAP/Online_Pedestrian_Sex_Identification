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
import random

from video import *
from data import *
from model import *
