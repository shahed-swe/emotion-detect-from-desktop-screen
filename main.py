import numpy as np
from PIL import ImageGrab
import cv2
import time
import win32gui, win32ui, win32con, win32api
from deepface import DeepFace
import cv2
import pywhatkit


def grab_screen(region=None):
    hwin = win32gui.GetDesktopWindow()
    if region:
            left,top,x2,y2 = region
            width = x2 - left + 1
            height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)
    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height,width,4)
    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())
    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)


def emoDecider(st1):
    frequency=[]
    for j in st1:
        count=0        
        for i in st1:
            
            if i==j:
                count=count+1
        frequency.append(count)
    print(frequency)
    
    for l in range(0,len(frequency)):
        if frequency[l]==max(frequency):
            emoIndex=l
    return emoIndex


def main():
    last_time = time.time()
    emotionList=[]
    t_end=time.time()+(60*0.5)
    face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    text2 = ""
    


    while time.time()<=t_end:
        # 1920 windowed mode
        screen = grab_screen(region=(0,40,1920,1120))
        img = cv2.resize(screen,None,fx=0.4,fy=0.3)

        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.1,4)


        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+w), (255,0,0),2)
            try:
                result = DeepFace.analyze(img, actions = ['emotion'])
                text2 = result['dominant_emotion'].upper()
                emotionList.append(str(text2))
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img,text2,
                        (40,180),
                        font,1,
                        (65,105,225),
                        2,cv2.LINE_4)
            except:
                pass
            

        cv2.imshow('img',img)
        k=cv2.waitKey(30) & 0xff
        if k==27:
            break
    cv2.destroyAllWindows ()


    finalLI=[]
    for i in emotionList:
        if i !='NEUTRAL':
            finalLI.append(i)       
        
    print(finalLI)
    
    

    pywhatkit.playonyt(finalLI[emoDecider(finalLI)])


main()