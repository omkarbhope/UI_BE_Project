import cv2
# from model import FacialExpressionModel
import dlib
import numpy as np

cap = cv2.VideoCapture(0)

def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)

def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

make_1080p()
change_res(1920, 1080)

# facec = cv2.CascadeClassifier('E://Youtube/Real-Time-Face-Expression-Recognition/haarcascade_frontalface_default.xml')
# model = FacialExpressionModel("E://Youtube/Real-Time-Face-Expression-Recognition/model.json", "E://Youtube/Real-Time-Face-Expression-Recognition/model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C://Users/omkar/Downloads/shape_predictor_68_face_landmarks.dat")
predictor2 = dlib.shape_predictor("E://shape_predictor_194_face_landmarks.dat")

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FPS, 30)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self,red,blue,green,pigment=0.3):
        _, fr = self.video.read()
        fr = cv2.flip(fr,1)
        output = fr.copy()
        output2 = fr.copy()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        # faces = facec.detectMultiScale(gray_fr, 1.3, 5)
        faces2 = detector(gray_fr)

        def l(i):
            return (landmarks.part(i).x,landmarks.part(i).y)

        def li(i):
            return (landmarks2.part(i).x,landmarks2.part(i).y)

        for face in faces2:

            landmarks = predictor(gray_fr, face)
            landmarks2 = predictor2(gray_fr, face)
            
            for n in range(0, 68):
                pts_array = np.array([l(54),l(55),l(56),l(57),l(58),l(59),l(48),l(60),l(67),l(66),l(65),l(64)],np.int32)
                pts_array = pts_array.reshape((-1,1,2))
                cv2.fillPoly(fr,[pts_array],(blue,green,red))
                pts_array2 = np.array([l(48),l(49),l(50),l(51),l(52),l(53),l(54),l(64),l(63),l(62),l(61),l(60)],np.int32)
                pts_array2 = pts_array2.reshape((-1,1,2))
                cv2.fillPoly(fr,[pts_array2],(blue,green,red))
                dist_x = li(37)[0] - li(33)[0]
                dist_x = int(dist_x)
                newPt = (li(37)[0]+dist_x,li(33)[1])
                pts_array3 = np.array([newPt,li(33),li(37)])
                pts_array3 = pts_array3.reshape((-1,1,2))
                cv2.fillPoly(output2,[pts_array3],(0,0,0))
                cv2.line(output2, li(27), li(28), (0,0,0), 1)
                cv2.line(output2, li(28), li(29), (0,0,0), 1)
                cv2.line(output2, li(29), li(30), (0,0,0), 1)
                cv2.line(output2, li(30), li(31), (0,0,0), 2)
                cv2.line(output2, li(31), li(33), (0,0,0), 2)
                cv2.line(output2, li(33), li(34), (0,0,0), 2)
                cv2.line(output2, li(34), li(35), (0,0,0), 2)
                cv2.line(output2, li(35), li(36), (0,0,0), 2)
                cv2.line(output2, li(36), li(37), (0,0,0), 3)

                dist_x2 = li(59)[0] - li(55)[0]
                dist_x2 = int(dist_x2)
                newPt2 = (li(59)[0]+dist_x2,li(55)[1])
                pts_array4 = np.array([newPt2,li(55),li(59)])
                pts_array4 = pts_array4.reshape((-1,1,2))
                cv2.fillPoly(output2,[pts_array4],(0,0,0))
                cv2.line(output2, li(59), li(58), (0,0,0), 3)
                cv2.line(output2, li(58), li(57), (0,0,0), 3)
                cv2.line(output2, li(57), li(56), (0,0,0), 2)
                cv2.line(output2, li(56), li(55), (0,0,0), 2)
                cv2.line(output2, li(55), li(53), (0,0,0), 2)
                cv2.line(output2, li(53), li(52), (0,0,0), 1)
                cv2.line(output2, li(52), li(51), (0,0,0), 1)
                cv2.line(output2, li(51), li(50), (0,0,0), 1)
                cv2.line(output2, li(50), li(49), (0,0,0), 1)


        # for (x, y, w, h) in faces:
        #     fc = gray_fr[y:y+h, x:x+w]

        #     roi = cv2.resize(fc, (48, 48))
        #     pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

        #     cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
        #     cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.addWeighted(output2, 0.7, output, 1 - 0.7,0, output)
        cv2.addWeighted(fr, pigment, output, 1 - 0.3,0, output)
        

        _, jpeg = cv2.imencode('.jpg', output)
        return jpeg.tobytes()
