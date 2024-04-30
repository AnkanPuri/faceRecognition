import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime
 
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Unable to open camera.")
    # Handle the error (e.g., exit the program or use a different camera index)


 
Ankan_image = face_recognition.load_image_file("photos/Ankan.jpg")
Ankan_encoding = face_recognition.face_encodings(Ankan_image)[0]
Aviral_image = face_recognition.load_image_file("photos/Aviral.jpg")
Aviral_encoding = face_recognition.face_encodings(Aviral_image)[0]
Ankan2_image = face_recognition.load_image_file("photos/Ankan2.jpg")
Ankan2_encoding = face_recognition.face_encodings(Ankan2_image)[0]
Shubh_image = face_recognition.load_image_file("photos/Shubh.jpg")
Shubh_encoding = face_recognition.face_encodings(Shubh_image)[0]
Akash_image = face_recognition.load_image_file("photos/Akash.jpg")
Akash_encoding = face_recognition.face_encodings(Akash_image)[0]
Ayansh_image = face_recognition.load_image_file("photos/Ayansh.jpg")
Ayansh_encoding = face_recognition.face_encodings(Ayansh_image)[0]
 
known_face_encoding = [
Ankan_encoding,
Aviral_encoding,
Ankan2_encoding,
Shubh_encoding,
Akash_encoding,
Ayansh_encoding,

]
 
known_faces_names = [
"Ankan",
"Aviral",
"Ankan",
"Shubh",
"Darkness",
"Ayansh",
]
 
students = known_faces_names.copy()
 
face_locations = []
face_encodings = []
face_names = []
s=True
 
 
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
 
 
 
f = open(current_date+'.csv','w+',newline = '')
lnwriter = csv.writer(f)
 
while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    # rgb_small_frame = small_frame[:,:,::-1]
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding, tolerance=0.8)

            name=""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]
 
            face_names.append(name)
            if name in known_faces_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10,100)
                fontScale              = 1.5
                fontColor              = (255,0,0)
                thickness              = 3
                lineType               = 2
                
 
                cv2.putText(frame,name+' Present', 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
 
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])
    cv2.imshow("attendence system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
video_capture.release()
cv2.destroyAllWindows()
f.close()
