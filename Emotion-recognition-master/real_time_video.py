from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import serial
import time
import math
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

conn = pymysql.connect(host='springboot-db.cc58omnt7fw3.ap-northeast-2.rds.amazonaws.com',port=3306,user='admin',password='qwer1234',db='qna',charset='utf8')
cur = conn.cursor()

## id와 password, 그리고 확인 버튼의 UI를 만드는 부분
# tk.Label(window, text = "Username : ").grid(row = 0, column = 0, padx = 10, pady = 10)
# tk.Label(window, text = "Password : ").grid(row = 1, column = 0, padx = 10, pady = 10)
# tk.Entry(window, textvariable = user_id).grid(row = 0, column = 1, padx = 10, pady = 10)
# tk.Entry(window, textvariable = password, show='*').grid(row = 1, column = 1, padx = 10, pady = 10)
# tk.Button(window, text = "Login", command =check_data).grid(row = 2, column = 1, padx = 10, pady = 10)
# window.resizable(False,False)
# window.mainloop()

sql = "INSERT INTO question(subject,content,author_id,created_date,hit_count) VALUES ('test', 'test', 1, now(), 0)"
cur.execute(sql);
sql = "SELECT LAST_INSERT_ID()"
cur.execute(sql)
result = cur.fetchall()
bno = result[0][0]
print(bno)            
Check = 0 
new = 0
dict = {} 
list = [0,0,0] 

# 라즈베리파이 환경
# ser = serial.Serial('/dev/ttyACM0', 9600)
## 컴퓨터 환경
ser = serial.Serial('COM5', 9600)
# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.83-0.82.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" , "happy", "neutral"]

# starting video streaming
# cv2.namedWindow('your_face')
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
                        dict[emotion] = prob
                        
                        # i가 2일 때라는 것은 모든 감정에 대한 예측 값을 딕셔너리에 넣었다는 뜻.   
                        if ( i == 2 ):
                            # 밑에 3줄은 지금 몇 초 지났는지에 대한 것을 result에 갱신
                            end = time.time()
                            result = end-begin
                            result = round(result,1)
                            # 이 과정을 2초 동안 할 것이기 때문에 계속해서 result 값이 2초인지 확인
                            if (math.ceil(result) == 2.0):
                                # 2초를 초기화 하기 위한 Check 값 0으로 설정
                                Check = 0
                                # 2초동안 어떤 감정을 했는 지에 대한 값을 저장하기 위해 list의 최댓값을 max_value에 저장
                                max_value = max(list)
                                # 만약에 list의 최댓값이 angry일 때
                                if (list.index(max_value) == 0): # angry로 시리얼 통신
                                    if ser.readable() :
                                        sum = list[0] + list[1] + list[2]
                                        list = [list[i]/sum * 100  for i in range(3)]
                                        print(list)
                                        val = 'angry'
                                        sql = "INSERT INTO emotion(question_id, author_id,created_date,angry,happy,neutral) values('%s',1,now(),%f,%f,%f)" %(bno,list[0],list[1],list[2])
                                        # sql = "INSERT INTO chart(bno,commenter,angry,happy,neutral) VALUES ('%s','%s',%f,%f,%f)" %(bno,user_id.get(),list[0],list[1],list[2])
                                        cur.execute(sql)
                                        val = val.encode('utf-8')
                                        ser.write(val)
                                        
                                        print("Atomize TURNED ON")
                                    else: continue
                                    print("자신의 감정은 angry입니다.")
                                    list = [0,0,0]
                                # 만약에 list의 최댓값이 happy일 때
                                elif (list.index(max_value) == 1): # happy로 시리얼 통신
                                    if ser.readable() :
                                        sum = list[0] + list[1] + list[2]
                                        list = [list[i]/sum * 100  for i in range(3)]
                                        print(list)
                                        val = 'happy'
                                        sql = "INSERT INTO emotion(question_id, author_id,created_date,angry,happy,neutral) values('%s',1,now(),%f,%f,%f)" %(bno,list[0],list[1],list[2])
                                        # sql = "INSERT INTO chart(bno,commenter,angry,neutral,happy) VALUES ('%s','%s',%f,%f,%f)" %(bno,user_id.get(),list[0],list[1],list[2])
                                        cur.execute(sql)
                                        val = val.encode('utf-8')
                                        ser.write(val)
                                        # print("Atomize")
                                    else: continue
                                    print("자신의 감정은 happy입니다.")
                                    list = [0,0,0]
                                # 만약에 list의 최댓값이 neutral일 때
                                elif (list.index(max_value) == 2): # neutral로 시리얼 통신
                                    if ser.readable() :
                                        sum = list[0] + list[1] + list[2]
                                        list = [list[i]/sum * 100  for i in range(3)]
                                        print(list)
                                        val = 'neutral'
                                        sql = "INSERT INTO emotion(question_id, author_id,created_date,angry,happy,neutral) values('%s',1,now(),%f,%f,%f)" %(bno,list[0],list[1],list[2])
                                        # sql = "INSERT INTO chart(bno,commenter,angry,neutral,happy) VALUES ('%s','%s',%f,%f,%f)" %(bno,user_id.get(),list[0],list[1],list[2])
                                        cur.execute(sql)
                                        val = val.encode('utf-8')
                                        ser.write(val)
                                        print("Atomize TURNED OFF")
                                    else: continue
                                    print("자신의 감정은 neutral입니다.")
                                    list = [0,0,0]
                            # 혹시 error 가 생겨 설정한 시간 2초 이상이 넘어 갈 경우
                            # 다시 2초를 초기화 하기 위한 변수 재설정
                            elif (result > 2):
                                Check = 0
                            # 딕셔너리의 저장된 value 값의 최댓값(어떤 감정인지)을 찾아 Count를 해주어 list에 저장
                            # list[0] 은 angry, list[1] 은, happy, list[2] 는 neutral의 Count 이다.
                            if max(dict,key=dict.get) == 'angry':
                                list[0] = list[0] + 1
                            elif max(dict,key=dict.get) == 'happy':
                                list[1] = list[1] + 1
                            elif max(dict,key=dict.get) == 'neutral':
                                list[2] = list[2] + 1

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

            # cv2.imshow('your_face', frameClone)
            # cv2.imshow("Probabilities", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                
                break
except KeyboardInterrupt:
    pass

conn.commit()
conn.close()
camera.release()
cv2.destroyAllWindows()