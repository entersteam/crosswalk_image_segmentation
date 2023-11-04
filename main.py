from ultralytics import YOLO
import cv2
import serial 

model_YOLO = YOLO("./best.pt") #가중치 파일 경로

crosswalk_mask = cv2.imread('./mask.bmp', cv2.IMREAD_GRAYSCALE)

cap = cv2.VideoCapture(0)

arduino = serial.Serial('COM4', 9600, timeout=1)

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
    else:
        signal = '0'
    arduino.write(signal.encode())
    
    pass
