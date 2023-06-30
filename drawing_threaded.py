# THREADED Draw

from __future__ import print_function
import datetime
from imutils.video import WebcamVideoStream
import cv2
import numpy as np
import time
import imutils
import os
import queue
import threading
import HandTrackingModule as htm


def saveFile(inv_img):
    base_filename = "signtest"
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = base_filename + '_' + timestamp + '.jpg'
    cv2.imwrite(filename, inv_img)


cap = WebcamVideoStream(src=0).start()

def process_frame(out_q: queue.Queue, cap: WebcamVideoStream,):
    detector = htm.handDetector()

    brushThickness = 5
    eraserThickness = 200
    pTime = 0
    cTime = 0

    # folderPath = "Header"
    # myList = os.listdir(folderPath)
    # print(myList)
    # overlayList = []
    # for imPath in myList:
    #     image = cv2.imread(f'{folderPath}/{imPath}')
    #     overlayList.append(image)
    # print(len(overlayList))
    # print(overlayList)
    # header = overlayList[0]
    drawColor = (0, 0, 255)
    temptime = 0

    save_img = cv2.imread(r"C:\Users\vishn\OneDrive\Desktop\python projects\Handrecog_proj\icons8-save-96.png")
    save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)

    delete_img = cv2.imread(r"C:\Users\vishn\OneDrive\Desktop\python projects\Handrecog_proj\delete.png")
    delete_img = cv2.resize(delete_img, (96,96))
    delete_img = cv2.cvtColor(delete_img,cv2.COLOR_RGB2BGR)
    # ret, mask = cv2.threshold(save_img, 1, 255, cv2.THRESH_BINARY)


    detector = htm.handDetector()
    xp, yp = 0, 0

    imgCanvas = np.zeros((480, 640, 3), np.uint8)

    while True:
        #1. import img
        img = cap.read()
        img  = imutils.resize(img,width=640, height= 480)
        img = cv2.flip(img, 1)
        #2. find hand landmark
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList)!= 0:
            #print(lmList)
            #tip of middle and index finger position
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]


            #3.check 2 finger up
            fingers = detector.fingersUp()
            # print(fingers)
            #4.if if seletion mode if 2 fnger is up
            if fingers[1] and fingers[2]:
                xp, yp = 0, 0
                # print("selection Mode")
                #checking fr the click
                if y1 < 125:
                    if 0 < x1 < 125:
                        saveFile(imgInv)
                        print("save")
                    elif (520 <x1<520+96):
                        imgCanvas[0:480,0:640,:] = np.zeros((480, 640,3),dtype=np.uint8)
                

                cv2.rectangle(img, (x1, y1-25), (x2, y2+25), drawColor, cv2.FILLED)
            #5.if drawing mode when finger is up
            
            if fingers[1] and fingers[2] == False:
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                # print("Drawing Mode")
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                if drawColor == (0, 0, 0):
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                else:
                    temptime = time.time()
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                    

                xp, yp = x1, y1
        #to change BGR to GRAY scale and Again Gray scale to BGR and converting the image to bitwise
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # Setting the header image
        # img[0:75, 0:640] = header
        # cv2.imshow("test",header)
        
        added_image_save = cv2.addWeighted(img[0:96,0:96,:],0,save_img[0:96,0:96,:],1,0)
        img[20:20+96,20:20+96] = added_image_save
        # print(added_image_save.shape)

        # print(time.time() - temptime)
    
        added_image_delete = cv2.addWeighted(img[0:96,0:96,:],0,delete_img[0:96,0:96,:],1,0)
        # print(added_image_delete.shape)
        # print(img[540:540+96 , 20:20+96].shape)
        img[20:20+96 , 540:540+96] = added_image_delete
        
        # cv2.imshow("test",img)
        #img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
        out_q.put_nowait([img, imgCanvas, imgInv])

def display_frames(out_q: queue.Queue):
    """Display the frames"""
    work = True
    cv2.namedWindow("Pr_Creation")
    cv2.namedWindow("Canvas")
    cv2.namedWindow("Inverse")
    while work:
        try:
            size = out_q.qsize()
            
            print(size)
            if size == 0:
                continue
            data = out_q.get()
            cv2.imshow("Pr_Creation", data[0])
            #Canvas of the screen
            cv2.imshow("Canvas", data[1])
            #
            cv2.imshow("Inverse", data[2])

            cv2.waitKey(1)
        except (TimeoutError, queue.Empty):
            work = False
            continue

# Create the shared queue and launch both threads
cross_threads = queue.Queue()
t1 = threading.Thread(
    target=process_frame,
    args=(cross_threads, cap),
)
t2 = threading.Thread(
    target=display_frames,
    args=(cross_threads,),
)

t1.start()
time.sleep(2)
t2.start()

cross_threads.join()

cap.stop()
cv2.destroyAllWindows()