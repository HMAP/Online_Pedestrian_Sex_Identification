from dependency import *
class video:
    def __init__(self,model):
        print "Hey There... Now we'll start"
        self.load(model)

    def load(self,model):
        def GetInterest(imgname,name=1):
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
                vec.append(ang)

            return np.asarray(points),np.asarray(vec)

        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier('assets/haarcascade_frontalface_alt.xml')
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        count = 0
        while True:
            flag, frame = cap.read()
            if count != 24:
                if flag:
                    # The frame is ready and already captured
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    for (x,y,w,h) in faces:
                        face = gray[y:y+h, x:x+w]
                        dim=cv2.resize(face, (100,100), interpolation = cv2.INTER_AREA)
                        dim = np.array(dim)
                        # dim=dim.flatten()
                        # print dim
                        I1x,I1y,I1x2,I1y2,Int1 = GetInterest(face,1)
                        Int1,vec1 = vectorise(I1x,I1y,I1x2,I1y2,Int1)
                        transform = model.pca.transform([vec1.flatten()])[0]
                        # print transform.shape
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
                    # pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    # print str(pos_frame)+" frames"
                else:
                    # The next frame is not ready, so we try to read it again
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
                    print "frame is not ready"
                    # It is better to wait for a while for the next frame to be ready
                    cv2.waitKey(1000)
                count = 0
            else:
                count += 1
            if cv2.waitKey(10) == 27:
                break
            # if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            #     # If the number of captured frames is equal to the total number of frames,
            #     # we stop
            #     break

def plot_gallery(images, titles, h, w, n_row=5, n_col=15):
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
