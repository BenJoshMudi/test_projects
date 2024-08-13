import cv2
import face_recognition
import os
import pickle
import firebase_admin
from firebase_admin import credentials, db, storage
from pymongo import MongoClient
from flask import Flask, jsonify, request
from datetime import datetime
import streamlit as st
from PIL import Image
import io
import threading
import requests

# Firebase and MongoDB Setup
cred = credentials.Certificate("C:/Users/mudia/OneDrive/Desktop/Face_Recognition/Images/serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendancerealtime-c1ea8-default-rtdb.firebaseio.com/",
    'storageBucket': "faceattendancerealtime-c1ea8.appspot.com"
})
client = MongoClient("mongodb+srv://<username>:<password>@cluster0.mongodb.net/<dbname>?retryWrites=true&w=majority")
db_mongo = client['attendance_management']
attendance_collection = db_mongo['attendance']

# Importing student images into a list
ImagefolderPath = "C:/Users/mudia/OneDrive/Desktop/Face_Recognition/Images"
PathList = os.listdir(ImagefolderPath)
imgList = []
studentids = []

for path in PathList:
    imgList.append(cv2.imread(os.path.join(ImagefolderPath, path)))
    studentids.append(os.path.splitext(path)[0])
    fileName = f'{ImagefolderPath}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)

# Function to generate all the encodings
def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

print("Encoding Started ...")
encodingListKnown = findEncodings(imgList)
encodingListKnownwithids = [encodingListKnown, studentids]
print("Encoding Complete")

file = open("EncodeFile.p", 'wb')
pickle.dump(encodingListKnownwithids, file)
file.close()
print("File Saved")

# Initialize Flask app
app = Flask(__name__)

@app.route('/recognize', methods=['POST'])
def recognize():
    image_file = request.files['image']
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(encodingListKnown, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(encodingListKnown, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = studentids[best_match_index]
        face_names.append(name)

        # Log attendance to MongoDB
        if name != "Unknown":
            record = {"name": name, "time": datetime.now()}
            attendance_collection.insert_one(record)

        # Log attendance to Firebase
        ref = db.reference(f'students/{name}')
        student_info = ref.get()
        ref.update({
            "total_attendance": student_info['total_attendance'] + 1,
            "last_attendance_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    return jsonify({"status": "success", "recognized_faces": face_names})

def run_flask():
    app.run(host='0.0.0.0', port=5000)

# Streamlit Frontend
def run_streamlit():
    st.title("Facial Recognition Attendance System")

    uploaded_image = st.file_uploader("Upload an image for recognition", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        response = requests.post("http://localhost:5000/recognize", files={"image": img_byte_arr})
        response_data = response.json()

        if response_data['status'] == "success":
            st.write(f"Recognized Faces: {', '.join(response_data['recognized_faces'])}")
        else:
            st.write("Recognition failed.")

# Running Flask in a separate thread
flask_thread = threading.Thread(target=run_flask)
flask_thread.start()

# Running Streamlit
run_streamlit()
