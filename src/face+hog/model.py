from dependency import *
class Model:
    def __init__(self,Data,n):
        self.load(Data,n)

    def load(self,Data,n):
        data = []
        data_set = []
        data_set_labels = []
        def GetInterest(imgname):
            img = imgname
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.float32(img)
            #cv2.imshow(imgname + "ori",img)

            Ix=np.zeros(img.shape)
            Iy=np.zeros(img.shape)
            kernelx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
            Ix = cv2.filter2D(img,-1,kernelx)
            Iy = cv2.filter2D(img,-1,np.transpose(kernelx))

            Ix2 = Ix*Ix
            Iy2 = Iy*Iy
            Ixy = Ix*Iy

            t=np.zeros(img.shape)
            intpoints = []
            # print(threshold)
            # print img.shape
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if (i/5==0):
                        if(j/5==0):
                            intpoints.append([i,j])

            return Ix,Iy,Ix2,Iy2,np.array(intpoints)

        def vectorise(Ix, Iy, Ix2, Iy2, Int):
            H = np.sqrt(Ix2 + Iy2)
            angles = np.arctan2(Iy, Ix) /np.pi
            w = Ix.shape[0]
            h = Ix.shape[1]
            points = []
            window=4
            vec = []
            off = 0
            # print Int
            for p in Int:
                ang = np.zeros(128)
                points.append(p)
                for i in range(0,4):
                    for j in range(0,4):
                        for x in range(0,window):
                            for y in range(0,window):
                                block = 4*i  + j
                                indx = p[0] + (i-2)*window + x
                                indy = p[1] + (j-2)*window + y
                                lamb = angles[indx][indy] + 1
                                if (lamb>=0 and lamb<0.25): off = 0
                                elif (lamb>=0.25 and lamb<0.5): off = 1
                                elif (lamb>=0.5 and lamb<0.75): off = 2
                                elif (lamb>=0.75 and lamb<1): off = 3
                                elif (lamb>=1 and lamb<1.25): off = 4
                                elif (lamb>=1.25 and lamb<1.5): off = 5
                                elif (lamb>=1.5 and lamb<1.75): off = 6
                                elif (lamb>=1.75 and lamb<2): off = 7
                                ang[block*8 + off] += H[indx][indy]
                # print ang
                vec.append(ang)

            return np.asarray(points),np.asarray(vec)

        # print len(Data.males)

        for person in Data.males:
            I1x,I1y,I1x2,I1y2,Int1 = GetInterest(person)
            # print Int1
            Int1,vec1 = vectorise(I1x,I1y,I1x2,I1y2,Int1)
            data.append([vec1.flatten(),"male"])

        for person in Data.females:
            I1x,I1y,I1x2,I1y2,Int1 = GetInterest(person)
            Int1,vec1 = vectorise(I1x,I1y,I1x2,I1y2,Int1)
            data.append([vec1.flatten(),"female"])
        random.seed(datetime.now().microsecond)
        random.shuffle(data)
        for tup in data:
            data_set.append(tup[0])
            data_set_labels.append(tup[1])
        X_train, X_test, y_train, y_test = train_test_split(data_set, data_set_labels, test_size=0.3, random_state=datetime.now().second)
        # n_components = n
        # self.pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
        # X_train_pca = self.pca.transform(X_train)
        # X_test_pca = self.pca.transform(X_test)
        s=[]
        self.clf = LinearSVC()
        self.clf.fit(X_train,y_train)
        s.append(self.clf.score(X_test,y_test))

        self.clf = RandomForestClassifier()
        self.clf.fit(X_train,y_train)
        s.append(self.clf.score(X_test,y_test))

        self.clf = Perceptron()
        self.clf.fit(X_train,y_train)
        s.append(self.clf.score(X_test,y_test))

        self.clf = AdaBoostClassifier()
        self.clf.fit(X_train,y_train)
        s.append(self.clf.score(X_test,y_test))

        print s
        # print "Train Score ",s
        # print "Model succesfully Built."
