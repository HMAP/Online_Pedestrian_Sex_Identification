from dependency import *
class Data:
    def __init__(self, cut = 1):
        self.load(cut)

    def load(self,cut):

        face_cascade = cv2.CascadeClassifier('assets/haarcascade_frontalface_alt.xml')
        self.males = []
        t='malestaff'
        for male in listdir('data/faces94/'+t):
            for file in listdir('data/faces94/'+t+'/'+male)[:cut]:
                # print file
                gray = Image.open('data/faces94/'+t+'/'+male+'/'+file,'r')
                gray=gray.convert('L')
                gray=np.asarray(gray)
                # cv2.imshow('face',gray)
                # cv2.waitKey(127)
                try:
                    (x,y,w,h) = face_cascade.detectMultiScale(gray, 1.3, 5)[0]
                    face = gray[y:y+h, x:x+w]
                    dim=cv2.resize(face, (100,100), interpolation = cv2.INTER_AREA)
                    self.males.append(np.array(dim))
                except:
                    pass
                    # print 'male',file

        self.females = []
        for female in listdir('data/faces94/female'):
            for file in listdir('data/faces94/female/'+female)[:cut]:
                gray = Image.open('data/faces94/female/'+female+'/'+file,'r')
                gray=gray.convert('L')
                gray=np.asarray(gray)
                try:
                    (x,y,w,h) = face_cascade.detectMultiScale(gray, 1.3, 5)[0]
                    face = gray[y:y+h, x:x+w]
                    dim=cv2.resize(face, (100,100), interpolation = cv2.INTER_AREA)
                    self.females.append(np.array(dim))
                except:
                    pass

        for file in listdir('data/nottingham/male'):
            gray = Image.open('data/nottingham/male/'+file,'r')
            gray=gray.convert('L')
            gray=np.asarray(gray)
            try:
                (x,y,w,h) = face_cascade.detectMultiScale(gray, 1.3, 5)[0]
                face = gray[y:y+h, x:x+w]
                dim=cv2.resize(face, (100,100), interpolation = cv2.INTER_AREA)
                self.males.append(np.array(dim))
                # cv2.imwrite('data/nottingham/cropped/male/'+file+'.jpg', dim)
            except:
                pass
                # print 'male',file

        for file in listdir('data/nottingham/female'):
            gray = Image.open('data/nottingham/female/'+file,'r')
            gray=gray.convert('L')
            gray=np.asarray(gray)
            try:
                (x,y,w,h) = face_cascade.detectMultiScale(gray, 1.3, 5)[0]
                face = gray[y:y+h, x:x+w]
                dim=cv2.resize(face, (100,100), interpolation = cv2.INTER_AREA)
                self.females.append(np.array(dim))
                # cv2.imwrite('data/nottingham/cropped/female/'+file+'.jpg', dim)
            except:
                pass
                # print 'female',file
        print "Total Males:",len(self.males)
        print "Total females:", len(self.females)
