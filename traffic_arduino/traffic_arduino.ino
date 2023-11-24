int ledPin = 13;

void setup() {
  pinMode(ledPin, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    if (command == '1') { // main.py에서의 시리얼 통신
      digitalWrite(ledPin, HIGH); //LED를 킴.
      Serial.println("LED ON");
    } else if (command == '0') {
      digitalWrite(ledPin, LOW); // LED를 끔.
      Serial.println("LED OFF");
    }
  }
}
