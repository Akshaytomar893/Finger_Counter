import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self , mode=False , maxHands=2 , detectionCon=0.5, trackingCon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.trackingCon=trackingCon

        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(self.mode , self.maxHands, self.detectionCon , self.trackingCon)
        self.mpDraw=mp.solutions.drawing_utils


    def findHands(self , img , draw=True):
        
        img_rgb=cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks:
            for handLandmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img , handLandmarks , self.mpHands.HAND_CONNECTIONS)
        return img
    

    def findPosition(self , img , handNo=0 , draw=False):

        lm_list=[]
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]
            for id, landmark in enumerate(myHand.landmark):
                h,w,c=img.shape
                cx,cy=int(landmark.x*w) , int(landmark.y*h)
                lm_list.append([id,cx,cy])
                if draw:
                    cv2.circle(img , (cx,cy) , 8 , (255,0,0) , cv2.FILLED)

        return lm_list

def main():
    cTime=0
    pTime=0
    capture=cv2.VideoCapture(0)
    detector=handDetector()
    while True:
        success , img=capture.read()
        img=detector.findHands(img)
        lm_list=detector.findPosition(img)
        cTime=time.time()
        fps=(1/(cTime-pTime))
        pTime=cTime



        cv2.putText(img ,str(int(fps)) , (20,40) , cv2.FONT_HERSHEY_PLAIN,3 , (255,0,0) ,3,)
        cv2.imshow("Video" , img)
        if cv2.waitKey(1) & 0xFF==27:
            break
    cv2.destroyAllWindows()



if __name__=="__main__":
    main()