import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import deque
import math


class SubmarineLineTracker:
    def __init__(self, model_path, video_path=None, camera_id=0):
        self.model_path = model_path
        self.video_path = video_path
        self.camera_id = camera_id

        # Kamera ayarları
        self.IMAGE_WIDTH = 640
        self.IMAGE_HEIGHT = 480

        # Çizgi takip parametreleri - Optimize edilmiş
        self.DEAD_ZONE = 30  # Kodunuzdaki threshold değeri
        self.MIN_AREA = 100  # Minimum alan eşiği (düşürüldü)
        self.MAX_AREA = 50000  # Maksimum alan eşiği

        # Anomali tespit parametreleri
        self.ANGLE_THRESHOLD = 45  # Derece cinsinden maksimum açı değişimi
        self.DISTANCE_THRESHOLD = 100  # Piksel cinsinden maksimum konum değişimi

        # Hareket yumuşatma için geçmiş veriler
        self.position_history = deque(maxlen=10)
        self.angle_history = deque(maxlen=5)

        # İstatistikler
        self.stats = {
            'frames_processed': 0,
            'cable_detected': 0,
            'anomalies_detected': 0,
            'avg_distance': 0
        }

        # YOLO modeli yükle
        try:
            self.yolo_model = YOLO(model_path)
            print(f"YOLO modeli başarıyla yüklendi: {model_path}")
        except Exception as e:
            print(f"YOLO modeli yüklenirken hata: {e}")
            self.yolo_model = None

    def initialize_camera(self):
        """Kamera veya video dosyasını başlatır"""
        if self.video_path:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise ValueError(f"Video dosyası açılamadı: {self.video_path}")
        else:
            cap = cv2.VideoCapture(self.camera_id)
            if not cap.isOpened():
                raise ValueError(f"Kamera açılamadı: {self.camera_id}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.IMAGE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.IMAGE_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)

        return cap

    def detect_cable_with_yolo(self, frame):
        """YOLO ile kablo tespiti yapar"""
        if self.yolo_model is None:
            return frame, False

        try:
            results_list = self.yolo_model.predict(source=frame, show=False, verbose=False)
            results = results_list[0]
            cable_detected = False

            for detection in results.boxes:
                x_min, y_min, x_max, y_max, conf, _ = detection.data[0].tolist()
                label = results.names[int(detection.cls[0])]

                if conf >= 0.7 and label == 'Kablo':
                    cable_detected = True
                    # Tespit edilen bölgeyi işaretle
                    cv2.rectangle(frame, (int(x_min), int(y_min)),
                                  (int(x_max), int(y_max)), (0, 255, 0), 2)
                    cv2.putText(frame, f"Kablo: {conf:.2f}",
                                (int(x_min), int(y_min) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            return frame, cable_detected
        except Exception as e:
            print(f"YOLO tespit hatası: {e}")
            return frame, False

    def detect_line_advanced(self, frame):
        """Gelişmiş çizgi tespit algoritması - Optimize edilmiş morfolojik temizleme"""
        # HSV renk uzayına çevir
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Siyah renk için HSV aralığı (kodunuzdaki değerler)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 107])  # Üst sınır artırıldı

        # Siyah maske oluştur
        black_mask = cv2.inRange(hsv, lower_black, upper_black)

        # Gaussian blur uygula (kodunuzdaki gibi)
        blurred_mask = cv2.GaussianBlur(black_mask, (9, 9), 0)

        # Büyük kernel ile morfolojik temizleme (kodunuzdaki gibi)
        kernel = np.ones((9, 9), np.uint8)
        cleaned_mask = cv2.morphologyEx(blurred_mask, cv2.MORPH_OPEN, kernel)

        # Ek olarak CLOSE operasyonu ekleyelim
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

        return cleaned_mask

    def find_line_center_and_angle(self, mask, frame):
        """Çizginin merkez noktasını ve açısını bulur"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None, None, frame

        # En büyük konturun alanını kontrol et
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area < self.MIN_AREA or area > self.MAX_AREA:
            return None, None, None, frame

        # Merkez noktasını hesapla
        M = cv2.moments(largest_contour)
        if M['m00'] == 0:
            return None, None, None, frame

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # Çizginin açısını hesapla (fitLine kullanarak)
        angle = None
        if len(largest_contour) >= 5:
            try:
                [vx, vy, x, y] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
                angle = math.degrees(math.atan2(vy, vx))

                # Açı çizgisini görselleştir
                rows, cols = frame.shape[:2]
                lefty = int((-x * vy / vx) + y)
                righty = int(((cols - x) * vy / vx) + y)
                cv2.line(frame, (cols - 1, righty), (0, lefty), (255, 0, 0), 2)

            except:
                pass

        # Konturun etrafına çerçeve çiz
        cv2.drawContours(frame, [largest_contour], -1, (0, 255, 255), 2)

        # Merkez noktasını işaretle
        cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
        cv2.circle(frame, (cx, cy), 15, (255, 255, 255), 2)

        return cx, cy, angle, frame

    def detect_anomalies(self, cx, cy, angle):
        """Anomali tespiti yapar"""
        anomalies = []

        if len(self.position_history) > 0:
            last_pos = self.position_history[-1]
            distance_change = math.sqrt((cx - last_pos[0]) ** 2 + (cy - last_pos[1]) ** 2)

            if distance_change > self.DISTANCE_THRESHOLD:
                anomalies.append(f"Ani konum değişimi: {distance_change:.1f}px")

        if angle is not None and len(self.angle_history) > 0:
            angle_changes = [abs(angle - prev_angle) for prev_angle in self.angle_history]
            if any(change > self.ANGLE_THRESHOLD for change in angle_changes):
                anomalies.append(f"Ani açı değişimi: {max(angle_changes):.1f}°")

        # Geçmişe ekle
        self.position_history.append((cx, cy))
        if angle is not None:
            self.angle_history.append(angle)

        return anomalies

    def calculate_navigation_command(self, cx, cy, frame_center_x, frame_center_y, angle=None):
        """Navigasyon komutunu hesaplar"""
        if cx is None or cy is None:
            return "Çizgi bulunamadı", 0, 0

        # Yatay sapma
        horizontal_error = cx - frame_center_x
        distance = abs(horizontal_error)

        # Yumuşatılmış kontrol
        if len(self.position_history) >= 3:
            recent_positions = list(self.position_history)[-3:]
            avg_x = sum(pos[0] for pos in recent_positions) / len(recent_positions)
            horizontal_error = avg_x - frame_center_x

        # Navigasyon kararı
        if abs(horizontal_error) <= self.DEAD_ZONE:
            command = "Düz git"
            turn_strength = 0
        elif horizontal_error < -self.DEAD_ZONE:
            command = "Sola dön"
            turn_strength = min(abs(horizontal_error) / 100, 1.0)  # 0-1 arası
        else:
            command = "Sağa dön"
            turn_strength = min(abs(horizontal_error) / 100, 1.0)

        # Açı bilgisi varsa bunu da dikkate al
        angle_correction = 0
        if angle is not None:
            # Çizgi çok eğikse ek düzeltme
            if abs(angle) > 15:
                angle_correction = angle / 45.0  # Normalize et

        return command, turn_strength, angle_correction

    def draw_ui_elements(self, frame, command, turn_strength, angle_correction,
                         distance, fps, anomalies, cable_detected):
        """Kullanıcı arayüzü elementlerini çizer"""
        h, w = frame.shape[:2]

        # Merkez çizgileri
        cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
        cv2.line(frame, (0, h // 2), (w, h // 2), (255, 255, 255), 1)

        # Ölü bölge
        cv2.rectangle(frame,
                      (w // 2 - self.DEAD_ZONE, 0),
                      (w // 2 + self.DEAD_ZONE, h),
                      (0, 255, 255), 1)

        # Bilgi paneli
        info_y = 30
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        info_y += 25
        color = (0, 255, 0) if command == "Düz git" else (0, 165, 255)
        cv2.putText(frame, f"Komut: {command}", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        info_y += 25
        cv2.putText(frame, f"Dönüş gücü: {turn_strength:.2f}", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        info_y += 20
        cv2.putText(frame, f"Mesafe: {distance:.1f}px", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        info_y += 20
        status = "AKTIF" if cable_detected else "PASİF"
        color = (0, 255, 0) if cable_detected else (0, 0, 255)
        cv2.putText(frame, f"YOLO: {status}", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Anomali uyarıları
        if anomalies:
            for i, anomaly in enumerate(anomalies):
                cv2.putText(frame, f"ANOMALI: {anomaly}", (10, h - 60 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # İstatistikler (sağ üst köşe)
        stats_x = w - 200
        cv2.putText(frame, f"İşlenen: {self.stats['frames_processed']}",
                    (stats_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"Tespit: {self.stats['cable_detected']}",
                    (stats_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"Anomali: {self.stats['anomalies_detected']}",
                    (stats_x, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def run(self):
        """Ana çalışma döngüsü"""
        try:
            cap = self.initialize_camera()
            print("Sistem başlatıldı. Çıkış için 'q' tuşuna basın...")

            while True:
                start_time = time.time()

                # Frame yakala
                ret, frame = cap.read()
                if not ret:
                    print("Frame yakalanamadı")
                    break

                original_frame = frame.copy()
                frame_center_x = frame.shape[1] // 2
                frame_center_y = frame.shape[0] // 2

                # YOLO ile kablo tespiti
                frame, cable_detected = self.detect_cable_with_yolo(frame)

                # Çizgi tespiti
                mask = self.detect_line_advanced(original_frame)
                cx, cy, angle, frame = self.find_line_center_and_angle(mask, frame)

                # Navigasyon hesaplama
                command = "Çizgi bulunamadı"
                turn_strength = 0
                angle_correction = 0
                distance = 0
                anomalies = []

                if cx is not None and cy is not None:
                    distance = math.sqrt((cx - frame_center_x) ** 2 + (cy - frame_center_y) ** 2)
                    command, turn_strength, angle_correction = self.calculate_navigation_command(
                        cx, cy, frame_center_x, frame_center_y, angle)

                    # Anomali tespiti
                    anomalies = self.detect_anomalies(cx, cy, angle)

                    # İstatistikleri güncelle
                    self.stats['cable_detected'] += 1 if cable_detected else 0
                    self.stats['anomalies_detected'] += len(anomalies)
                    self.stats['avg_distance'] = (self.stats['avg_distance'] * 0.9 + distance * 0.1)

                self.stats['frames_processed'] += 1

                # FPS hesapla
                fps = 1.0 / (time.time() - start_time)

                # UI elementlerini çiz
                self.draw_ui_elements(frame, command, turn_strength, angle_correction,
                                      distance, fps, anomalies, cable_detected)

                # Sonucu göster
                cv2.imshow("Denizaltı Çizgi Takip Sistemi", frame)
                cv2.imshow("Maske", mask)

                # Çıkış kontrolü
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):  # Screenshot
                    cv2.imwrite(f"screenshot_{int(time.time())}.jpg", frame)
                    print("Ekran görüntüsü kaydedildi")

        except Exception as e:
            print(f"Hata oluştu: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"Sistem durduruldu. Toplam işlenen frame: {self.stats['frames_processed']}")


def main():
    # Konfigürasyon
    model_path = "C:/Users/user/OneDrive/Masaüstü/best.pt"
    video_path = "C:/Users/user/OneDrive/Masaüstü/line.avi"

    # Tracker'ı başlat
    tracker = SubmarineLineTracker(model_path=model_path, video_path=video_path)
    tracker.run()


if __name__ == "__main__":
    main()