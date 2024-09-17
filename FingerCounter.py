import cv2
import time
import os
import HandTrackingModule as hm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = 'FingerImages'
myList = os.listdir('FingerImages')
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath})
    overlayList.append(image)

print(len(overlayList))
pTime = 0

detector = hm.handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        fingers = []

        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)  # Thumb is open
        else:
            fingers.append(0)  # Thumb is closed


        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)  # Finger is open
            else:
                fingers.append(0)  # Finger is closed

        totalFingers = fingers.count(1)
        print(f"Total fingers raised: {totalFingers}")

        if totalFingers <= len(overlayList):
            overlay_resized = cv2.resize(overlayList[totalFingers - 1], (200, 200))
            img[0:200, 0:200] = overlay_resized

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
