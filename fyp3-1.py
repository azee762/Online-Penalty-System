import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cvlib as cv
from PIL import Image, ImageOps
import os.path
from csv import writer
import cv2
import face_recognition
import numpy as np
import glob
# import PySimpleGUI as sg
from time import strftime


# Gender detection model
gender_detection_model = load_model('Models/gender_detection.model')
gender_detection_classes = ['male', 'female']

# Object detection model
configPath = 'Models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'Models/frozen_inference_graph.pb'

ObjectNames = []
ObjectNamesFile = 'Models/coco.names'

with open(ObjectNamesFile, 'rt') as f:
    ObjectNames = f.read().rstrip('\n').split('\n')

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Dress detection models
dress_labels = ['not allowed', 'allowed']
np.set_printoptions(suppress=True)
female_dress_model = load_model('Models/female_dress_model.h5', compile=False)
male_dress_model = load_model('Models/male_dress_model.h5', compile=False)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


# Female dress check
def female_dress_check(dress):
    image = cv2.cvtColor(dress, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = female_dress_model.predict(data)
    i = np.argmax(prediction)
    return dress_labels[i]


# Male dress check
def male_dress_check(dress):
    image = cv2.cvtColor(dress, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = male_dress_model.predict(data)
    i = np.argmax(prediction)
    return dress_labels[i]


# Crop head from body
def crop_top(person):
    haar_cascade = cv2.CascadeClassifier("Models/haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
    face = haar_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in face:
        body_box = 0, y + h, person.shape[1], person.shape[0]
    return body_box


# Predict gender
def predict_gender(face):
    face_crop = cv2.resize(face, (96, 96))
    face_crop = face_crop.astype("float") / 255.0
    face_crop = img_to_array(face_crop)
    face_crop = np.expand_dims(face_crop, axis=0)
    conf = gender_detection_model.predict(face_crop)[0]
    idx = np.argmax(conf)
    return gender_detection_classes[idx]


# Detect face
def detect_face(person):
    face, confidence = cv.detect_face(person)
    for idx, points in enumerate(face):
        face_box = points
    return face_box


# Encode images for face recognition
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


# Recognize by face
def get_id(person):
    imgS = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)
        if faceDis[matchIndex] < 0.50:
            rollNum = FaceIDs[matchIndex]
        else:
            rollNum = 'X'
    return rollNum, faceLoc


# Report student
def report(rollNum):
    print("fun called")
    file = "Reports/" + strftime("%d-%b-%Y") + ".csv"
    if not os.path.isfile(file):
        temp = open(file, "w")
        write_data_to_csv(file, ["Roll Number", "Name", "Fine", "Time", "Image ID"])
    res = add_fine(file, rollNum)
    print(res)
    load_fine_names(file)
    return res


# Add fine to student
def add_fine(file, rollNum):
    res = check_reg(file, rollNum)
    if not res:
        name = get_name(rollNum)
        if name != "N/A":
            time_now = strftime("%H:%M")
            data = [rollNum, name, FineAmount, time_now, index]
            write_data_to_csv(file, data)
    return res


# Check if student is registerd
def check_reg(file, rollNum):
    with open(file, 'r+') as f:
        myDataList = f.readlines()
        FineNameList = []
        for line in myDataList:
            entry = line.split(',')
            FineNameList.append(entry[0])
        if rollNum not in FineNameList:
            return False
        else:
            return True


# Get student name from roll number
def get_name(rollNum):
    if rollNum in rollNumList:
        index = rollNumList.index(rollNum)
        return nameList[index]
    else:
        return "N/A"


# Write student name and fine to csv file
def write_data_to_csv(file, data):
    with open(file, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(data)


# Save image of student
def save_img(body, face, rollNum):
    if rollNum != "X":
        file_name = "Captures/Person/Person-" + str(rollNum) + "-" + str(index) + ".jpg"
        file_fname = "Captures/Face/Face-" + str(rollNum) + "-" + str(index) + ".jpg"
        cv2.imwrite(file_name, body)
        cv2.imwrite(file_fname, face)
        return True
    else:
        return False


# Load saved roll number and names
def load_students():
    names = []
    rollNums = []
    with open('names.csv', 'r+') as f:
        FileData = f.readlines()
        for lines in FileData:
            entry = lines.split(',')
            rollNums.append(entry[0])
            names.append(entry[1].rstrip("\n"))
    return names, rollNums


# Load last saved index
def get_index():
    try:
        temp_data = []
        list_of_files = glob.glob('Reports/*')
        latest_file = max(list_of_files, key=os.path.getctime)
        with open(latest_file, 'r+') as f:
            FileData = f.readlines()
            for lines in FileData:
                entry = lines.split(',')
                temp_data.append(entry[-1])
        return int(temp_data[-1]) + 1
    except:
        return 1


# Load fined students from file of current date
def load_fine_names(file):
    names = []
    with open(file, 'r+') as f:
        FileData = f.readlines()
        for lines in FileData:
            entry = lines.split(',')
            if (entry[0] != "Roll Number"):
                temp = entry[0] + " " + entry[1]
                names.append(temp)


# Detect Person
def detect_person(frame):
    flag = False
    framecopy = frame.copy()
    classIds, confs, bbox = net.detect(frame, confThreshold=0.4)
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if confidence > 0.61 and classId == 1:
                x, y, w, h = box
                person = framecopy[y:y + h, x:x + w]
                try:
                    face_box = detect_face(person)
                    face = person[face_box[1]:face_box[3], face_box[0]:face_box[2]]
                    gender = predict_gender(np.copy(face))
                    body_box = crop_top(person)
                    body = person[body_box[1]:body_box[3], body_box[0]:body_box[2]]
                    if gender == "female":
                        cv2.rectangle(frame, (box[0] + face_box[0], box[1] + face_box[1]),
                                      (box[0] + face_box[2], box[1] + face_box[3]), (255, 0, 255), 2)
                        dress = female_dress_check(body)
                    elif gender == "male":
                        cv2.rectangle(frame, (box[0] + face_box[0], box[1] + face_box[1]),
                                      (box[0] + face_box[2], box[1] + face_box[3]), (255, 255, 0), 2)
                        dress = male_dress_check(body)

                    if dress == "allowed":
                        print("allowed")
                        cv2.rectangle(frame, (box[0] + body_box[0], box[1] + body_box[1]),
                                      (box[0] + body_box[2], box[1] + body_box[3]), (0, 255, 0), 2)
                    elif dress == "not allowed":
                        print("not allowed")
                        cv2.rectangle(frame, (box[0] + body_box[0], box[1] + body_box[1]),
                                      (box[0] + body_box[2], box[1] + body_box[3]), (0, 0, 255), 2)
                        rollNum, faceLoc = get_id(face)
                        cv2.rectangle(frame, (box[0] + face_box[0] + faceLoc[3], box[1] + face_box[1] + faceLoc[0]),
                                      (box[0] + face_box[0] + faceLoc[1], box[1] + face_box[1] + faceLoc[2]),
                                      (255, 0, 0), 2)
                        cv2.rectangle(frame,
                                      (box[0] + face_box[0] + faceLoc[3], box[1] + face_box[1] + faceLoc[2] + 35),
                                      (box[0] + face_box[0] + faceLoc[1], box[1] + face_box[1] + faceLoc[2]),
                                      (255, 255, 255), cv2.FILLED)
                        cv2.putText(frame, rollNum,
                                    (box[0] + face_box[0] + faceLoc[3] + 5, box[1] + face_box[1] + faceLoc[2] + 30),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
                        res = report(rollNum)
                        print(res)
                        if not res:
                            flag = save_img(person, face, rollNum)
                except:
                    pass

    return frame, flag


nameList = []
rollNumList = []

FaceDataPath = 'FaceData'
images = []
FaceIDs = []
myList = os.listdir(FaceDataPath)
for cl in myList:
    curImg = cv2.imread(f'{FaceDataPath}/{cl}')
    images.append(curImg)
    FaceIDs.append(os.path.splitext(cl)[0])
print("Loaded " + str(len(FaceIDs)) + " faces.")

nameList, rollNumList = load_students()

encodeListKnown = findEncodings(images)

FineAmount = "Rs. 500"
index = int(get_index())


# Video source
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    frame, flag = detect_person(frame)
    if flag:
        index += 1

    res = cv2.resize(frame, (1152, 648))
    cv2.imshow("Online Penalty System", res)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()