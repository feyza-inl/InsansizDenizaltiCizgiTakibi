import cv2
import numpy as np
import time
from ultralytics import YOLO

CAMERA_DEVICE_ID = 0
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
fps = 0


def set_camera_properties(cap, width, height):
    """Kamera özelliklerini ayarlar."""
    cap.set(3, width)
    cap.set(4, height)


def capture_frame(cap):
    """Bir kareyi kameradan yakalar."""
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Failed to read a frame from the camera")
    return frame


def detect_shapes_with_yolov8(frame, yolo_model):
    """YOLOv8 ile şekil tespiti yapar."""
    results_list = yolo_model.predict(source=frame, show=False)
    results = results_list[0]
    for detection in results.boxes:
        x_min, y_min, x_max, y_max, conf, _ = detection.data[0].tolist()
        label = results.names[int(detection.cls[0])]

        if conf >= 0.8:
            if label == 'Kablo':  # Kablo tespit edildiğinde işlem yap
                print("Kablo tespit edildi")
                frame = cv2.putText(frame, "Kablo", (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imwrite("Kablo.jpg", frame)
    return frame


def detect_black_area(frame):
    """Siyah alanları tespit eder ve merkez noktasını döner."""
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 70])  # Siyah renk aralığı

    black_mask = cv2.inRange(hsv_image, lower_black, upper_black)

    blurred_mask = cv2.GaussianBlur(black_mask, (9, 9), 0)
    kernel = np.ones((7, 7), np.uint8)  # Maskeyi temizle
    cleaned_mask = cv2.morphologyEx(blurred_mask, cv2.MORPH_OPEN, kernel)

    black_contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    black_cx, black_cy = 0, 0
    if black_contours:
        largest_black_contour = max(black_contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_black_contour)
        if area > 100:  # Alan büyüklüğünü kontrol et
            M = cv2.moments(largest_black_contour)
            if M['m00'] != 0:
                black_cx = int(M['m10'] / M['m00'])
                black_cy = int(M['m01'] / M['m00'])
                cv2.circle(frame, (black_cx, black_cy), 5, (0, 255, 0), -1)
    return black_cx, black_cy, frame


def calculate_distance_and_direction(black_cx, black_cy, frame_center_x, frame_center_y):
    """Kablonun orta noktası ile kameranın merkezi arasındaki mesafeyi hesaplar ve yön belirtir."""
    distance = np.sqrt((black_cx - frame_center_x) ** 2 + (black_cy - frame_center_y) ** 2)
    print(f"Distance: {distance}")

    if black_cx < frame_center_x - 30:
        print("Sola dön")
        return "Sola dön"
    elif black_cx > frame_center_x + 30:
        print("Sağa dön")
        return "Sağa dön"
    else:
        print("Düz git")
        return "Düz git"


def process_frame(frame, yolo_model):
    """Bir çerçeveyi işleyerek şekil tespiti yapar ve siyah alanı analiz eder."""
    frame_center_x = frame.shape[1] // 2
    frame_center_y = frame.shape[0] // 2

    # YOLO ile şekil tespiti yap
    frame = detect_shapes_with_yolov8(frame, yolo_model)

    # Siyah alanı tespit et
    black_cx, black_cy, frame = detect_black_area(frame)

    # Kablonun orta noktası ile kameranın orta noktası arasındaki mesafeyi hesapla
    direction = calculate_distance_and_direction(black_cx, black_cy, frame_center_x, frame_center_y)

    return frame, direction


def main():
    try:
        video_path = "C:/Users/user/Downloads/video1.mp4"
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Video açılamadı. Lütfen '{video_path}' dosyasının mevcut olduğundan emin olun.")
            exit()

        print("Press 'Esc' to exit...")

        fps = 0

        model_path = "C:/Users/user/OneDrive/Masaüstü/best.pt"
        yolo_model = YOLO(model_path)

        while True:
            start_time = time.time()

            frame = capture_frame(cap)

            # Frame üzerinde işlem yap
            frame, direction = process_frame(frame, yolo_model)

            # FPS hesapla
            end_time = time.time()
            seconds = end_time - start_time
            fps = 1.0 / seconds

            # FPS ve yönlendirme bilgisini ekranda göster
            cv2.putText(frame, f"FPS: {fps:.2f}", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, direction, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(e)

    finally:
        cv2.destroyAllWindows()
        cap.release()


if __name__ == "__main__":
    main()
