from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import serial
import configparser
import time
import math
import sys
import pymysql
# tkinter를 사용하기 위한 import
from distutils.cmd import Command
from tkinter import *
import tkinter as tk

# # 사용자 id와 password를 비교하는 함수
# def check_data():
#     sql = "select id, pwd from user_info where id = '%s' and pwd = '%s'" %(user_id.get(), password.get()) 
#     cur.execute(sql)
#     result = cur.fetchall()
#     if len(result) == 0  :
#         print("Check your Username/Password")
#     elif user_id.get() == result[0][0] and password.get() == result[0][1] :
#         print("Logged IN Successfully")
#         quit()
#     else:
#         print("Check your Username/Password")
        
# def quit():
# 	window.destroy()

# # tkinter 객체 생성
# window = Tk()

# 사용자 id와 password를 저장하는 변수 생성
# user_id, password = StringVar(), StringVar()

config = configparser.ConfigParser()
config.read('config.ini')
db_host = config['DB']['HOST']
db_user = config['DB']['USER']
db_pass = config['DB'].get("PASS", "default password")
db_name = config['DB'].get("NAME", False)
conn = pymysql.connect(host=db_host,port=3306,user=db_user,password=db_pass,db=db_name,charset='utf8')
cur = conn.cursor()

## id와 password, 그리고 확인 버튼의 UI를 만드는 부분
# tk.Label(window, text = "Username : ").grid(row = 0, column = 0, padx = 10, pady = 10)
# tk.Label(window, text = "Password : ").grid(row = 1, column = 0, padx = 10, pady = 10)
# tk.Entry(window, textvariable = user_id).grid(row = 0, column = 1, padx = 10, pady = 10)
# tk.Entry(window, textvariable = password, show='*').grid(row = 1, column = 1, padx = 10, pady = 10)
# tk.Button(window, text = "Login", command =check_data).grid(row = 2, column = 1, padx = 10, pady = 10)
# window.resizable(False,False)
# window.mainloop()

userId = sys.argv[1];
print("real_time userId : ",end ='')
print(userId)
# print("bno : " + bno);
quetionId = sys.argv[2];
print("real_time quetionId : ",end ='')
print(quetionId)  
Check = 0 
new = 0
dict = {} 
list = [0,0,0] 
cnt = 0;

# # 라즈베리파이 환경
# ser = serial.Serial('/dev/ttyACM0', 9600)
# 컴퓨터 환경
# ser = serial.Serial('COM5', 9600)
# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/mini_XCEPTION.83-0.82.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" , "happy", "neutral"]

# starting video streaming
cv2.namedWindow('your_face')
camera = cv2.VideoCapture(0)
try:
    while True:
            frame = camera.read()[1]
            #reading the frame
            frame = imutils.resize(frame,width=300)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
            
            canvas = np.zeros((250, 300, 3), dtype="uint8")
            frameClone = frame.copy()
            if len(faces) > 0:

                # 얼굴이 한명이라도 인식이 되었을 때 시간 시작
                # Check == 0 은 계속해서 2초동안 초기화를 위한 변수.
                if (Check == 0):
                    begin = time.time()
                    Check = Check + 1

                faces = sorted(faces, reverse=True,
                key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                (fX, fY, fW, fH) = faces
                            # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
                    # the ROI for classification via the CNN
                roi = gray[fY:fY + fH, fX:fX + fW]
                roi = cv2.resize(roi, (48, 48))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                
                
                preds = emotion_classifier.predict(roi)[0]
                emotion_probability = np.max(preds)
                label = EMOTIONS[preds.argmax()]
            else: continue

        
            # enumerate 는 list를 인덱스 번호와 값을 튜플(값이 변하지 않는) 형태로 주는 방식
            # zip 은 같은 인덱스 번호인 애들끼리 튜플 형태로 주는 방식
            # 그래서 i(0~2)는 인덱스 번호를 알려주고, emotion(0~2)은 3개 감정을 3개 저장, prob(0~1)은 감정에 대한 예측 값을 저장 
            # ex) 예를 들면 이런 식으로 값이 들어있으면 --> (0, (2, 0.6)) 
            # index 번호 0에다가 2번째 감정(neutral)에 0.6(60%) 예측 값  
            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                        # construct the label text
                        text = "{}: {:.2f}%".format(emotion, prob * 100)
                        
                        # 사람의 감정을 몇 초동안 파악을 해서 그 시간동안 이 사람의 감정이 무엇인지 찾기. 
                
                        # 딕셔너리(key와 value 값을 가진 것)에 emotion(0~2) key 값과 어떤 감정인지 예측 값 prob(0~1) value 값 저장
                        dict[emotion] = prob * 100;
                        if (i == 2):
                            # 밑에 3줄은 지금 몇 초 지났는지에 대한 것을 result에 갱신
                            end = time.time()
                            result = end-begin
                            result = round(result,1)
                            cnt = cnt + 1;
                            list[0] = list[0] + dict['angry'];
                            list[1] = list[1] + dict['happy'];
                            list[2] = list[2] + dict['neutral']; 
                            # 이 과정을 1초 동안 할 것이기 때문에 계속해서 result 값이 2초인지 확인
                            if (math.ceil(result) >= 2.0):
                                Check = 0
                                # 1초를 초기화 하기 위한 Check 값 0으로 설정
                                list[0] = list[0] / cnt;
                                list[1] = list[1] / cnt;
                                list[2] = list[2] / cnt;
                                angryText = "angry 수치 : ";
                                happyText = "happy 수치 : ";
                                neutralText = "neutral 수치 : ";
                                print(angryText, end='');
                                print(list[0]);
                                print(happyText, end='')
                                print(list[1]);
                                print(neutralText, end='')
                                print(list[2]);
                                Check = 0
                                sql = "INSERT INTO emotion(analysis_id, author_id,created_date,angry,happy,neutral) values('%s','%s',now(),%f,%f,%f)" %(quetionId,userId,list[0],list[1],list[2]);
                                cur.execute(sql)
                                list = [0,0,0]
                                cnt = 0;
                        

                        w = int(prob * 300)
                        cv2.rectangle(canvas, (7, (i * 35) + 5),
                        (w, (i * 35) + 35), (0, 0, 255), -1)
                        cv2.putText(canvas, text, (10, (i * 35) + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 2)
                        cv2.putText(frameClone, label, (fX, fY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                                    (0, 0, 255), 2)

            cv2.imshow('your_face', frameClone)
            cv2.imshow("Probabilities", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                
                break
except KeyboardInterrupt:
    pass

conn.commit()
conn.close()
camera.release()
cv2.destroyAllWindows()