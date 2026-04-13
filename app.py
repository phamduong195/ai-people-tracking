from ultralytics import YOLO
import cv2
from playsound import playsound
import threading

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# Tránh spam âm thanh
alert_on = False

def play_alert():
    global alert_on
    if not alert_on:
        alert_on = True
        playsound("alert.mp3")  # bạn cần file này
        alert_on = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    boxes = results[0].boxes

    people = []
    phones = []

    # Tách object
    for box in boxes:
        cls = int(box.cls[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls == 0:
            people.append((x1, y1, x2, y2))
        elif cls == 67:
            phones.append((x1, y1, x2, y2))

    # Check overlap
    danger = False

    for px1, py1, px2, py2 in people:
        for fx1, fy1, fx2, fy2 in phones:

            # nếu phone nằm trong người
            if fx1 > px1 and fx2 < px2 and fy1 > py1 and fy2 < py2:
                danger = True

                cv2.putText(frame, "WARNING: USING PHONE!",
                            (px1, py1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)

    # Phát âm thanh
    if danger:
        threading.Thread(target=play_alert).start()

    annotated = results[0].plot()

    cv2.imshow("AI Safety Camera", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()