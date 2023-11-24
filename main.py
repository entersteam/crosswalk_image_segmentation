from ultralytics import YOLO
import cv2
import serial
import os
import tensorflow as tf
import numpy as np
import torch

RED = (0,0,255)
GREEN = (0,255,0)

# 아두이노 포트 설정하기
PORT = 'COM4'

# 모델 불러오기
cw_model = YOLO("./cw_best.pt") # 횡단보도 모델
hm_model = YOLO("./hm_best.pt") # 사람 인식 모델

# 모델을 사용할 수 없을 때 임시로 마스크를 만드는 함수
def drawing_mask(event, x, y, flags, param):
    global mask_pts
    if event == cv2.EVENT_FLAG_LBUTTON:    
        mask_pts.append([x,y])
    if len(mask_pts) == 4:
        mask = np.full((h,w,1), 0, dtype=np.uint8)
        for i in [([0,1,2]),([0,2,3])]:
            mask = cv2.fillPoly(mask, [np.array(mask_pts, dtype = np.int32)[i]], (255,255,255))
        cv2.imwrite('mask.bmp', mask)
        cv2.destroyAllWindows()


if 'mask.bmp' in os.listdir(): #마스크파일이 존재하는지 확인
    crosswalk_mask = cv2.imread('./mask.bmp', cv2.IMREAD_GRAYSCALE)
else:
    cap = cv2.VideoCapture(0)
    _, img = cap.read()
    try:
        results = cw_model(img)
        print(results)
        for result in results:
            boxes = result.boxes.data
            clss = boxes[:, 5]

            indices = torch.where(clss == 0)
            masks = masks[indices]
            
            mask = torch.any(masks, dim=0).int() * 255
            cv2.imwrite('mask.bmp', mask.cpu().numpy())
    except:
        cv2.namedWindow('masking')
        h, w, _ = img.shape
        cv2.setMouseCallback("masking", drawing_mask)
        mask_pts = []
        
        cv2.imshow('masking', img)
        cv2.waitKey()
        crosswalk_mask = cv2.imread('./mask.bmp', cv2.IMREAD_GRAYSCALE)

# 포트로 시리얼 통신
arduino = serial.Serial(PORT, 9600, timeout=1) #포트에 따라 바꾸기


while True:
    human_existing = False
    ret, img = cap.read()
    if not ret:
        print('read error')
        break
    results = hm_model(img, stream=True)
    
    for i in results:
        boxes = i.boxes
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            point = ((x1+x2)/2, max(y1,y2)) # 사람이 있는 위치는 이미지 내 좌표로 변환
            try:
                if crosswalk_mask[point[1]][point[0]] != 0: # 사람의 좌표가 마스크이미지와 겹칠 경우 실행
                    human_existing = True
                    break
            except:
                pass
    
    # 사람이 존재하는지 작은 원으로 간단하게 표시
    if human_existing:
        signal = '1' # 불을 키는 신호
        circle_color = RED
    else:
        signal = '0' # 불을 끄는 신호
        circle_color = GREEN
    img = cv2.circle(img, (20,20), 13, circle_color, -1)
    
    # 신호를 인코딩하여 전송
    arduino.write(signal.encode())

    cv2.imshow('cam', img)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break
cv2.destroyAllWindows()