from flask import Flask, Response, render_template
from ultralytics import YOLO
import cv2
import math
import threading
from playsound import playsound

app = Flask(__name__)

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

track_history = {}
alert_on = False
danger_status = "SAFE"

def play_alert():
    global alert_on
    if not alert_on:
        alert_on = True
        playsound("alert.mp3")
        alert_on = False

def generate_frames():
    global danger_status

    while True:
        success, frame = cap.read()
        if not success:
            break

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

            if track_id in track_history:
                prev_cx, prev_cy = track_history[track_id]
                distance = math.hypot(cx - prev_cx, cy - prev_cy)

                if distance > 10:
                    moving = True

            track_history[track_id] = (cx, cy)

            for fx1, fy1, fx2, fy2 in phones:
                if fx1 > px1 and fx2 < px2 and fy1 > py1 and fy2 < py2:

                    if moving:
                        danger = True

                        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 3)
                        cv2.putText(frame, "WARNING!",
                                    (px1, py1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (0, 0, 255), 2)

        # cập nhật trạng thái
        if danger:
            danger_status = "DANGER"
            threading.Thread(target=play_alert).start()
        else:
            danger_status = "SAFE"

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return danger_status

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)