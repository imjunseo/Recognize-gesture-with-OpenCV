import cv2
import mediapipe as mp
import math

draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
hand = mp.solutions.hands()
hand2 = hand.Hands()

def point(x1, y1, x2, y2):
    return math.sqrt(math.pow(x1 - x2,2)) + math.sqrt(math.pow(y1 - y2,2))

compare = [[18,4],[6,8],[10,12],[14,16],[18,20]]
ref = [False,False,False,False,False]
gesture = [[True,True,True,True,True,"Hello!"],
           [False,True,True,False,False,"V!"],
           [False,False,True,False,False,"Fuck you!"],
           [False,False,False,False,True,"Good!"]]

while True:
    success,img = cap.read()
    h,w,c = img.shape
    RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = hand2.process(RGB)
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            for i in range(0.5):
                ref[i] = point(handLms.landmark[0].x,handLms.landmark[0].y,
                         handLms.landmark[compare[i][0]].x,handLms.landmark[compare[i][0]].y) < point(handLms.landmark[0].x,handLms.landmark[0].y,
                         handLms.landmark[compare[i][1]].x,handLms.landmark[compare[i][1]].y)
            print(ref)
            text_x = (handLms.landmark[0].x*w)
            text_y = (handLms.landmark[0].y*h)
            for i in range(0,len(gesture)):
                reg = True
                for j in range(0,5):
                    if(gesture[i][j] != ref[j]):
                        reg = False
                if(reg == True):
                    cv2.putText(img,gesture[i][5],(round(text_x) - 50,round(text_y) - 250),
                                cv2.FONT_HERSHEY_PLAIN,4,(0,0,0),4)
            draw.draw_landmarks(img,handLms,hand.HAND_CONNECTIONS)

    cv2.imshow("Gesture Recognition",img)
    cv2.waitKey(1)
                    
