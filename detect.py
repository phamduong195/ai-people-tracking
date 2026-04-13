from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# Line position
line_y = 250

# Lưu vị trí trước đó của mỗi ID
track_history = {}

# Counter
count_in = 0
count_out = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, classes=[0])
    annotated_frame = results[0].plot()

    # Vẽ line
    cv2.line(annotated_frame, (0, line_y), (640, line_y), (0, 0, 255), 2)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes
        ids = boxes.id.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = box
            center_y = int((y1 + y2) / 2)

            track_id = int(ids[i])

            # Lấy vị trí trước đó
            if track_id in track_history:
                prev_y = track_history[track_id]

                # Đi xuống (IN)
                if prev_y < line_y and center_y >= line_y:
                    count_in += 1

                # Đi lên (OUT)
                elif prev_y > line_y and center_y <= line_y:
                    count_out += 1

            # Update vị trí
            track_history[track_id] = center_y

    # Hiển thị count
    cv2.putText(annotated_frame, f"IN: {count_in}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.putText(annotated_frame, f"OUT: {count_out}",
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)

    cv2.imshow("Line Counting", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()