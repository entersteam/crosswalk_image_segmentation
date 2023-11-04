import serial
import time

# 아두이노와 시리얼 통신을 설정합니다. 포트 이름은 사용하는 컴퓨터에 따라 다를 수 있습니다.
arduino = serial.Serial('COM4', 9600, timeout=1)

# 사용자로부터 LED를 켤지 끌지 입력받습니다.
while True:
    command = input("LED를 켜려면 '1'을, 끄려면 '0'을 입력하세요: ")
    if command == '1' or command == '0':
        arduino.write(command.encode())  # 입력한 값을 아두이노로 보냅니다.
    else:
        print("올바른 명령을 입력하세요.")
    time.sleep(0.1)
