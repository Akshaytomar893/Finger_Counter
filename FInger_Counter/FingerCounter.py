import cv2
import os
import time
import HandTrackingModule as htm

wCam , hCam = 640 , 480
cap=cv2.VideoCapture(0)
cap.set(3 , wCam)
cap.set(4 , hCam)

img_folder_path="img_fingers"
mylist=os.listdir(img_folder_path)
overlayList=[]
for imPath in mylist:
    image = cv2.imread(f'{img_folder_path}/{imPath}')
    #print(f'{folderPath}/{imPath}')
    overlayList.append(image)

#print(len(overlayList))

detector=htm.handDetector(detectionCon=0.7)
tipIds=[4,8,12,16,20]
pTime=0
cTime=0
fps=0
while True:
    success , img=cap.read()
    img=detector.findHands(img)
    lmList=detector.findPosition(img )
    #print(lmList)
    if len(lmList) !=0:
        fingers=[]
        #thumb
        if lmList[16][1] > lmList[20][1]:
            if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
                fingers.append(1)
        elif lmList[16][1] < lmList[20][1]:
            if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]:
                fingers.append(1)
        else:
            fingers.append(0)
        
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        totalFingers=fingers.count(1)

        h , w , c=overlayList[totalFingers-1].shape
        img[0:h , 0:w]=overlayList[totalFingers-1]

        #cv2.rectangle(img, (20, 100), (100, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 15)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF==27:
            break
cv2.destroyAllWindows()

