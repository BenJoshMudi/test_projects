# FIRST STEP: BUILDING THE WEBCAM and THE GRAPHICS
# Boilerplate template code for running a webcam
import cv2
import os
import pickle
import cvzone
import numpy as np
import face_recognition 
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime

cred = credentials.Certificate("C:/Users/mudia/OneDrive/Desktop/Face_Recognition/Images/serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendancerealtime-c1ea8-default-rtdb.firebaseio.com/",
    'storageBucket': "faceattendancerealtime-c1ea8.appspot.com" 
})

bucket = storage.bucket()

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread("C:/Users/mudia/OneDrive/Desktop/Face_Recognition/Resources/background.png")

# Importing the mode images into a list
folderModePath = "C:/Users/mudia/OneDrive/Desktop/Face_Recognition/Resources/Modes"
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

#print(len(imgModeList))

# Load the encoding file
print("Loading the encode file ...")
file = open('C:/Users/mudia/OneDrive/Desktop/Face_Recognition/EncodeFile.p', 'rb')
encodingListKnownwithids = pickle.load(file)
file.close()
encodeListKnown, studentids = encodingListKnownwithids
#print(studentids)
print("Encode File Loaded ...")

modeType = 0
counter = 0
id = -1
imgstudent = []

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print('matches', matches)
        # print('faceDis', faceDis)

        matchindex = np.argmin(faceDis)
        # print("Match Index", matchindex)

        if matches[matchindex]:
            # print("Known Face Detected")
            # print(studentids[matchindex])

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 *4, y2 * 4, x1 * 4
            bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
            imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
            id = studentids[matchindex]

            if counter == 0:
                cvzone.putTextRect(imgBackground, "Loading....", (275, 400))
                cv2.imshow("Face Attendance", imgBackground)
                cv2.waitKey(1)
                counter = 1
                modeType = 1

    if counter != 0:
        # Getting data from our database
        if counter == 1:
            studentinfo = db.reference(f'students/{id}').get()
            print(studentinfo)

            # Get the image from the storage
            blob = bucket.get_blob(f'C:/Users/mudia/OneDrive/Desktop/Face_Recognition/Images/{id}.png')
            array = np.frombuffer(blob.download_as_string(), np.uint8)
            imgstudent = cv2.imdecode(array, cv2.IMREAD_COLOR)

            # Update data of attendance
            datetimeobject = datetime.strptime(studentinfo['last_attendance_time'],
                                                "%Y-%m-%d %H:%M:%S")
            secondsElapsed = (datetime.now()-datetimeobject).total_seconds()
            print(secondsElapsed)
            if secondsElapsed > 30:
                ref = db.reference(f'students/{id}')
                studentinfo['total_attendance'] +=1
                ref.child('total_attendance').set(studentinfo['total_attendance'])
                ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            else:
                modeType = 3
                counter = 0
                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
        
        if modeType != 3:
            if 10 < counter < 20:
                modeType = 2

            imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
                
            if counter <= 10:
                cv2.putText(imgBackground, str(studentinfo['total_attendance']), (861, 125),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                cv2.putText(imgBackground, str(studentinfo['Department']), (1006, 550),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(imgBackground, str(id), (1006, 493),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(imgBackground, str(studentinfo['standing']), (910, 625),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                cv2.putText(imgBackground, str(studentinfo['year']), (1025, 625),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                cv2.putText(imgBackground, str(studentinfo['starting_year']), (1125, 625),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

                (w, h), _ = cv2.getTextSize(studentinfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                offset = (414 - w) // 2
                cv2.putText(imgBackground, str(studentinfo['name']), (808 + offset, 445),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

                # Resize imgstudent to (216, 216) to fit the space on imgBackground
                imgstudent_resized = cv2.resize(imgstudent, (216, 216))

                # Place the resized image on imgBackground
                imgBackground[175:175+216, 909:909+216] = imgstudent_resized

            counter += 1

            if counter >= 20:
                counter = 0
                modeType = 0
                studentinfo = []
                imgstudent = []
                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    cv2.imshow("Webcam", img)
    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)
