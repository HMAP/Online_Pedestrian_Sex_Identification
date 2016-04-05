import cv2
import os
import numpy as np
class video:
    def __init__(self):
        print "Hey There... Now we'll start"
        self.load()

    def load(self):
        print os.listdir('./')
        cap = cv2.VideoCapture('./video.mp4')
        if not cap.isOpened():
            print "could not open"
            return
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        count = 0
        for jojo in range(0,length, 10):
            flag, frame = cap.read()
            if count != 24:
                if flag:
                    # The frame is ready and already captured
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
                    for (x, y, w, h) in rects:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.imshow('vid',frame)
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

v =video()
