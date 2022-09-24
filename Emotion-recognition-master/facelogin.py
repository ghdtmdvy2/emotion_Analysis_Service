import cv2
import numpy as np
import subprocess
import time
import pymysql

conn = pymysql.connect(host='springboot-db.cc58omnt7fw3.ap-northeast-2.rds.amazonaws.com',port=3306,user='admin',password='qwer1234',db='qna',charset='utf8')
cur = conn.cursor()
def countdown(num_of_secs):
    while num_of_secs:
        m, s = divmod(num_of_secs, 60)
        min_sec_format = '{:02d}:{:02d}'.format(m, s)
        print(min_sec_format, end='/r')
        time.sleep(1)
        num_of_secs -= 1

    print('Countdown finished.')


inp = 5
#countdown(inp)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = 'haarcascade_files/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

names = ['None', 'karina', 'winter', 'ningning', 'giselle']

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1980)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

minW = 0.1 * cam.get(cv2.CAP_PROP_FRAME_WIDTH)
minH = 0.1 * cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
i = 0

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(int(minW), int(minH))
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        if confidence < 50:
            id = names[id]
            i = i+1
        else:
            id = "unknown"

        confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)
    if i == 10: break
    if cv2.waitKey(1) > 0: break


sql = "select id from site_user where username='%s'" %(id);
cur.execute(sql);
result = cur.fetchall()
user_id = result[0][0]
print("face login users_id : ",end ='')
print(user_id)
sql = "INSERT INTO question(subject,content,author_id,created_date,hit_count) VALUES ('test', 'test', '%s', now(), 0)" %(user_id)
cur.execute(sql)
sql = "SELECT LAST_INSERT_ID()"
cur.execute(sql)
result = cur.fetchall()
questionId = result[0][0]
print("face login questionId : ",end ='')
print(questionId)
conn.commit()
conn.close()
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
subprocess.run(["python", "real_time_video.py", "%d" %(user_id), "%d" %(questionId)]) 