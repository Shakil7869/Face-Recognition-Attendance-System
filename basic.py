import cv2
import numpy as np
import face_recognition

imgbill = face_recognition.load_image_file("images basic/bill gates.jpg")
imgbill = cv2.cvtColor(imgbill, cv2.COLOR_BGR2RGB)
imgbill_test = face_recognition.load_image_file("images basic/bill gates test.jpg")
imgbill_test = cv2.cvtColor(imgbill_test, cv2.COLOR_BGR2RGB)

face_loc = face_recognition.face_locations(imgbill)[0]
encodeBill = face_recognition.face_encodings(imgbill)[0]
cv2.rectangle(imgbill,(face_loc[3],face_loc[0]),(face_loc[1],face_loc[2]), (255,0,255), 2)

face_test = face_recognition.face_locations(imgbill_test)[0]
encodeTest = face_recognition.face_encodings(imgbill_test)[0]
cv2.rectangle(imgbill_test, (face_test[3],face_test[0]), (face_test[1], face_test[2]), (255,0,255), 2)

result = face_recognition.compare_faces([encodeBill], encodeTest)
# lower distance is maximum match
face_dis = face_recognition.face_distance([encodeBill], encodeTest)
print(result, face_dis)
cv2.putText(imgbill_test,f'{result} {round(face_dis[0]),2}',(50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

cv2.imshow("bill gates", imgbill)
cv2.imshow("bill gates test", imgbill_test)
cv2.waitKey(0)

