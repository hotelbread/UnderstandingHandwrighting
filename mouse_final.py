#######################
### Mouse effect!!! ###
#######################

import tensorflow as tf
import cv2
import sys, os
import numpy as np
import time
from keras.models import load_model

oldx, oldy = None, None
pressed = False

#마우스를 이용해서 손글씨 숫자 입력받기
def on_mouse(event, x, y, flags, param):
    # event는 마우스 동작 상수값, 클릭, 이동 등등
    # x, y는 내가 띄운 창을 기준으로 좌측 상단점이 0,0이 됌
    # flags는 마우스 이벤트가 발생할 때 키보드 또는 마우스 상태를 의미, Shif+마우스 등 설정가능
    # param은 영상이룻도 있도 전달하고 싶은 데이타, 안쓰더라도 넣어줘야함
    global oldx, oldy, pressed# 밖에 있는 oldx, oldy 불러오기
    if event == cv2.EVENT_LBUTTONDOWN: # 왼쪽 클릭시 실행
        # oldx, oldy = x, y # 마우스가 눌렀을 때 좌표 저장, 띄워진 영상에서의 좌측 상단기준
        pressed = True
        print('EVENT_LBUTTONDOWN: %d, %d' % (x,y)) # 좌표출력

    elif event == cv2.EVENT_MOUSEMOVE: #마우스가 움직일때 발생
        if pressed == True:
            cv2.line(img, (oldx, oldy), (x, y), (0, 0, 0), 15, cv2.LINE_AA)
            #flags & cv2.EVENT_FLAG_LBUTTON: # ==를 쓰면 다른 키도 입력되었을 때 작동안하므로 & (and)사용
            # cv2.circle(img, (x, y), 5, (0, 255, 0), -1) # 단점이 빠르게 움직이면 끊김
            # circle은 끊기므로 line 이용
            # 마우스 클릭한 좌표에서 시작해서 마우스 좌표까지 그림
            #oldx, oldy = x, y # 그림을 그리고 또 좌표 저장   
            #cv2.line(img, (oldx, oldy), (x, y), (0, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('show me your number', img)

    elif event == cv2.EVENT_LBUTTONUP: # 왼쪽 클릭하고 뗏을때 발생
        pressed = False
        '''
        img_resized = cv2.resize(img, dsize=(28,28), interpolation=cv2.INTER_AREA)
        x_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        x_gray = cv2.bitwise_not(x_gray)
        # cv2.imshow('test', x_gray)
        xTest = x_gray.reshape(1, 784)
        #예측하기
        yTest = np.argmax(model.predict(xTest), axis=-1)
        print('it might be : ',yTest)
        '''
        print('EVENT_LBUTTONUP: %d, %d' % (x,y)) # 좌표출력

    oldx, oldy = x, y

#폴더생성 (같은 명의 폴더가 있을 경우 에러가 발생하지 않게해줌)
def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)

    except OSError:
        print('Error: Failed to create the directory.')

if __name__=='__main__':
    #모델 불러오기
    print('Now loading...')
    model = load_model('CNN_2layers_model5.h5')
    print('Now it is completed!!!')

    #흰색 컬러 영상 생성
    img = np.ones((28*8, 28*8, 3), dtype=np.uint8) * 255
    #img = np.ones((28, 28, 3), dtype=np.uint8) * 255
    #윈도우 창
    cv2.namedWindow('show me your number')
    #마우스 입력, namedWindow or imshow가 실행되어 창이 떠있는 상태에서만 사용가능
    #마우스 이벤트가 발생하면 on_mouse 함수 실행
    cv2.setMouseCallback('show me your number', on_mouse, img)
    #영상출력
    while True:
        cv2.imshow('show me your number', img)
        key = cv2.waitKey(1)
        if key == ord('c'):
            img = np.ones((28*8, 28*8, 3), dtype=np.uint8) * 255
            print('clear')
            cv2.destroyWindow('test')

        if key == ord('s'):
            #파일명을 시간으로 나타내기
            now = time
            filename = now.strftime('%y%m%d_%H%M')
            filefolder = now.strftime('%y%m%d')
            createDirectory(filefolder)
            print(filename)
            #영상저장
            cv2.imwrite(filefolder+'/' + filename + '.png',img)

        if key == ord('a'):
            #예측을 위해서 데이터 전처리
            print('img checking : ', img.shape)
            x_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            x_gray = cv2.bitwise_not(x_gray)
            print('bitwise_not checking : ', x_gray.shape)
            img_resize = cv2.resize(x_gray, dsize=(28,28), interpolation=cv2.INTER_AREA)
            cv2.imshow('test', img_resize)
            print('checking : ',img_resize.shape)
            x_test = img_resize.astype(np.float32)
            x_test = np.expand_dims(img_resize, axis=0)
            #x_test = np.reshape(img_resize, (28, 28, 1))
            print('expanding checking : ', x_test.shape)
            #예측하기
            y_test = np.argmax(model.predict(x_test), axis=-1)


            # y = model.predict_classes(x)
            print('it might be : ',y_test)

        if key == ord('q') or key == 27:
            print('See you!!!')
            break

    cv2.destroyAllWindows()
    print('end of mouse effect')
