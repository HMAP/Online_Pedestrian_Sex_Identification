import cv2
import os
from copy import deepcopy
import numpy as np
class video:
    def __init__(self):
        print "Hey There... Now we'll start"
        self.load()

    def load(self):
        print os.listdir('./')
        cap = cv2.VideoCapture('./2.mp4')
        if not cap.isOpened():
            print "could not open"
            return
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        print "FPS",fps
        print "length",length
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        count = 0
        back = 2
        prev = np.zeros((back,height,width),dtype=np.uint8)

        bga = np.zeros((height,width),dtype=np.uint8)
        pcount = 0

        while True:
            flag, frame = cap.read()
            # wo,ho=frame.shape
            if pcount >= back:
                pcount = 0
            if count != 24:
                if flag:
                    gray = np.array(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),dtype=np.uint8)
                    gray = np.array(cv2.GaussianBlur(gray,(3,3),0),dtype=np.uint8)
                    images = np.zeros(shape=gray.shape + (back,))
                    for popo in range(back):
                        images[:,:,popo] = prev[popo]
                    bga = np.array(np.mean(images, axis=2),dtype=np.uint8)
                    frameDelta = cv2.absdiff(bga,gray)
                    th=5
                    low = frameDelta < th
                    high = frameDelta > th
                    frameDelta[low] = 0
                    frameDelta[high] = 255
                    thresh = frameDelta
                    # thresh = cv2.dilate(frameDelta, None, iterations=1)
                    thresh = np.array(cv2.GaussianBlur(thresh,(3,3),0),dtype=np.uint8)
                    cv2.imshow('th',thresh)
                    prev[pcount]=gray
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
                    print "frame is not ready"
                    cv2.waitKey(1000)
                count = 0
            else:
                count += 1
            if cv2.waitKey(10) == 27:
                break
            pcount += 1
v =video()
