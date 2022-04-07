import numpy as numpy
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('C:\\Users\\Nik\\Desktop\\Python Proj\\facerecog\\src\\cascades\\data\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\\Nik\\Desktop\\Python Proj\\facerecog\\src\\cascades\\data\\haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('C:\\Users\\Nik\\Desktop\\Python Proj\\facerecog\\src\\cascades\\data\\haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("C:\\Users\\Nik\\Desktop\\Python Proj\\facerecog\\src\\cascades\\trainer.yml")

labels = {"person_name": 1}
with open("C:\\Users\\Nik\\Desktop\\Python Proj\\facerecog\\src\\cascades\\labels.pickle",'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1366)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

x = 1
while(True):
    #Capture frame
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=4)

    for (x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h,x:x+w] # (ycord_start, ycord_end)
        roi_color = frame[y:y+h,x:x+w]

        # recogniser using deep learned model prediction
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 55:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame, name, (x ,y - 10), font, 1, color, stroke, cv2.LINE_AA)
        else:
            name = "Unknown"
            print("Unknown")
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame, name, (x ,y - 10), font, 1, color, stroke, cv2.LINE_AA)

        # ADD MORE DATASET (Capture latest image if opencv detected facial)

        #img_item = "C:\\Users\\Nik\\Desktop\\Python Proj\\facerecog\\src\\cascades\\my-image.png"
        #cv2.imwrite(img_item, roi_gray)

        #img_item = "C:\\Users\\Nik\\Desktop\\Python Proj\\facerecog\\src\\cascades\\images\\jattapon-wat\\"        #File path for new image captured to be placed
        #type_file = ".png"
        #img_all = img_item + str(x) + type_file
        #cv2.imwrite(img_all , roi_gray)
        #x += 1

        color = (255,0,0) # BGR
        stroke = 2
        end_cord_x = x+w
        end_cord_y = y+h
        cv2.rectangle(frame, (x,y), (end_cord_x,end_cord_y),color, stroke)

        #subitems = eye_cascade.detectMultiScale(roi_gray)
        #for (ex, ey, ew, eh) in subitems:
        #    cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh),(0,255,0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()