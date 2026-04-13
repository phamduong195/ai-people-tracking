from ultralytics import YOLO
import cv2
from playsound import playsound
import threading
import math

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# Lưu vị trí trước đó
track_history = {}

# Tránh spam âm thanh
alert_on = False

def play_alert():
    global alert_on
    if not alert_on:
        alert_on = True
        playsound("alert.mp3")
        alert_on = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # TRACK để lấy ID
    results = model.track(frame, persist=True)

    boxes = results[0].boxes

    people = []
    phones = []

    if boxes.id is not None:
        ids = boxes.id.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy()

        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = map(int, box)
            cls = int(classes[i])
            track_id = int(ids[i])

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if cls == 0:
                people.append((track_id, x1, y1, x2, y2, cx, cy))
            elif cls == 67:
                phones.append((x1, y1, x2, y2))

    danger = False

    for person in people:
        track_id, px1, py1, px2, py2, cx, cy = person

        moving = False

        # Check movement
        if track_id in track_history:
            prev_cx, prev_cy = track_history[track_id]

            distance = math.hypot(cx - prev_cx, cy - prev_cy)

            if distance > 10:  # threshold di chuyển
                moving = True

        # Update vị trí
        track_history[track_id] = (cx, cy)

        # Check phone trong người
        for fx1, fy1, fx2, fy2 in phones:
            if fx1 > px1 and fx2 < px2 and fy1 > py1 and fy2 < py2:

                if moving:
                    danger = True

                    cv2.putText(frame, "DANGER: USING PHONE WHILE WALKING!",
                                (px1, py1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)

    if danger:
        threading.Thread(target=play_alert).start()

    annotated = results[0].plot()

    cv2.imshow("AI Safety Camera", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()