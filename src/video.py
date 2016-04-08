import cv2
from classifiers import *
import numpy as np
from imutils.object_detection import non_max_suppression
class video:
    def __init__(self,model):
        print "Hey There... Now we'll start"
        self.load(model)
    def load(self,model):
        cap = cv2.VideoCapture('../asd.mov')
        # face_cascade = cv2.CascadeClassifier('assets/haarcascade_frontalface_alt.xml')
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        count = 0
        while True:
            flag, frame = cap.read()
            if count != 24:
                if flag:
                    frame=cv2.resize(frame, (500,500), interpolation = cv2.INTER_AREA)
                    # The frame is ready and already captured
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    (rects, weights) = hog.detectMultiScale(gray, winStride=(2, 2), padding=(2, 2), scale=1.9)
                    # rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
                    # pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
                    for (x,y,w,h) in rects:
                        if h > 150 or w>100 or h < 100 or w < 50 or (x in range(0,500) and y in range(250,500)):
                            continue
                        body = gray[y:y+h, x:x+w]
                        dim=cv2.resize(body, (64,128), interpolation = cv2.INTER_AREA)
                        des = hog.compute(dim)
                        prediction = model.clf.predict([des.flatten()])[0]
                        print prediction
                        if prediction == 1:
                            colo = (255,0,0)
                            print 'male'
                        else:
                            colo = (0,0,255)
                            print "female"
                        cv2.rectangle(frame,(x,y),(x+w,y+h),colo,2)
                    cv2.imshow('video', frame)
                    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
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
a=Data()
m=Model(a)
v=video(m)
