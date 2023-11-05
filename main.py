from ultralytics import YOLO
import cv2
import serial
import os
import tensorflow as tf

RED = (0,0,255)
GREEN = (0,255,0)

model = tf.keras.Model
model.load_weights('./weights.pt')

model_YOLO = YOLO("./best.pt") #가중치 파일 경로

cap = cv2.VideoCapture(0)

def mouse_event(event, x, y, flags, param):
    global mask_pts
    if event == cv2.EVENT_FLAG_LBUTTON:    
        mask_pts.append([x,y])
    if len(mask_pts) == 4:
        mask = np.full((h,w,1), 0, dtype=np.uint8)
        for i in [([0,1,2]),([0,2,3])]:
            mask = cv2.fillPoly(mask, [np.array(mask_pts, dtype = np.int32)[i]], (255,255,255))
        cv2.imwrite('mask.bmp', mask)

if 'mask.bmp' in os.listdir():
    crosswalk_mask = cv2.imread('./mask.bmp', cv2.IMREAD_GRAYSCALE)
else:
    _, img = cap.read()
    h, w, c = img.shape
    cv2.namedWindow('masking')
    cv2.setMouseCallback("masking", mouse_event)
    mask_pts = []
    
    cv2.imshow('masking', img)
    cv2.waitKey()
    crosswalk_mask = cv2.imread('./mask.bmp', cv2.IMREAD_GRAYSCALE)
    
arduino = serial.Serial('COM4', 9600, timeout=1) #포트에 따라 바꾸기

while True:
    human_existing = False
    ret, img = cap.read()
    if not ret:
        print('read error')
        break
    results = model_YOLO(img, stream=True)
    
    for i in results:
        boxes = i.boxes
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            point = ((x1+x2)/2, max(y1,y2))
            if crosswalk_mask[point[1]][point[0]]:
                human_existing = True
                break
    
    if human_existing:
        signal = '1'
        circle_color = RED
    else:
        signal = '0'
        circle_color = GREEN
    img = cv2.circle(img, (630,630), 7, circle_color, -1)
    arduino.write(signal.encode())
    
    cv2.imshow('cam', img)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break
cv2.destroyAllWindows()
