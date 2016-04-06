from dependency import *
class video:
    def __init__(self,model):
        print "Hey There... Now we'll start"
        self.load(model)

    def load(self,model):
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
                        dim=dim.flatten()
                        # print dim
                        transform = model.pca.transform([dim])[0]
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
