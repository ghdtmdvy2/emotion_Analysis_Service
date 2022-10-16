import cv2
import numpy as np
import subprocess
import time
import pymysql
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
db_host = config['DB']['HOST']
db_user = config['DB']['USER']
db_pass = config['DB'].get("PASS", "default password")
db_name = config['DB'].get("NAME", False)
conn = pymysql.connect(host=db_host,port=3306,user=db_user,password=db_pass,db=db_name,charset='utf8')
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

names = ['None', 'HongSeungPyo', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
         '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
         '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36',
         '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48',
         '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60',
         '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72',
         '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84',
         '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96',
         '97', '98', '99', '100']

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

print(id);
sql = "select id from site_user where username='%s'" %(id);
# sql = "select id from site_user where username='admin'";
cur.execute(sql);
result = cur.fetchall();
print(result);
user_id = result[0][0]
print("face login users_id : ",end ='')
print(user_id)
sql = "INSERT INTO analysis(subject,content,author_id,created_date,hit_count) VALUES ('test', 'test', '%s', now(), 0)" %(user_id)
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