import numpy as np
from math import atan2
import Image
import cv2
class Descriptor:
    def __init__(self,imgFile):
        img = cv2.imread(imgFile)
        self.img_data = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        self.WIDTH = len(self.img_data[0])
        self.HEIGHT = len(self.img_data)
        self.gradientMatrix = [[0 for i in range(self.WIDTH)] for j in range(self.HEIGHT)]
        self.binWIDTH = self.WIDTH/4
        self.binHEIGHT = self.HEIGHT/4
        self.get_Gradient()
        self.des = self.get_Descriptor()
    def get_Gradient(self):
		self.Ix,self.Iy = np.gradient(self.img_data)
		for y in range(self.HEIGHT):
			for x in range(self.WIDTH):
				dy=float(self.Iy[y][x])
				dx=float(self.Ix[y][x])
				self.gradientMatrix[y][x]=[dx,dy,np.linalg.norm([dx,dy]),atan2(dy,1.0*dx)]
    # def get_Histogram(self,gradientMatrixTrimmed):
    #     weightSum = 0
    #     for y in range(self.self.binHEIGHT):
	# 		for x in range(self.binWIDTH):
	# 			weightSum=weightSum+gradientMatrixTrimmed[x][y][2]
    #     weights = [[gradientMatrixTrimmed[x][y][2]/weightSum for i in range(self.binWIDTH)] for j in range(self.binHEIGHT)]
    #     thetas  = [[gradientMatrixTrimmed[x][y][3] for x in range(self.binWIDTH)] for j in range(self.binHEIGHT)]
    #     bins	= [0,0.25*np.pi,0.5*np.pi,0.75*np.pi,1.0*np.pi,1.25*np.pi,1.5*np.pi,1.75*np.pi,2.0*np.pi]
    #     histogram = np.histogram(thetas,bins=bins,weights=weights)[0]
    #     return	histogram

    def get_Descriptor(self):
        des = []
        for y in range(0,self.HEIGHT,self.binHEIGHT):
            for x in range(0,self.WIDTH,self.binWIDTH):
                #print x,y
                weights=[]
                thetas =[]
                weightSum=0
                for j in range(y,y+self.binHEIGHT):
                    for  i in range(x,x+self.binWIDTH):
                        weight=self.gradientMatrix[j][i][2]
                        #print x,y,i,j,weight
                        weights.append(weight)
                        weightSum+=weight
                        thetas.append(self.gradientMatrix[j][i][3])
                bins	= [0,0.25*np.pi,0.5*np.pi,0.75*np.pi,1.0*np.pi,1.25*np.pi,1.5*np.pi,1.75*np.pi,2.0*np.pi]
                histogram = np.histogram(thetas,bins=bins,weights=weights)[0]
                #print weightSum
                histogram = histogram/weightSum
                print histogram
                des.append(histogram)
        des =np.array(des)
        return des
