import cv2
import numpy as np
import math
import time
import threading
from pymavlink import mavutil
import sys
import signal
import subprocess
import os
from datetime import datetime

# YOLO imports - eğer kurulu değilse hata vermez
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
    print("✅ YOLO kütüphanesi yüklendi")
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠ YOLO kütüphanesi bulunamadı, anomali tespiti devre dışı")

print("⏳ Sistem başlatılıyor...")
time.sleep(10)

# Graceful shutdown için
shutdown_flag = False
camera_active = False


def signal_handler(sig, frame):
    global shutdown_flag, camera_active
    print('\n🛑 Kapatma sinyali alındı, güvenli kapatılıyor...')
    shutdown_flag = True
    camera_active = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# --- Pixhawk bağlantısı kur ---
def baglanti_kur(port="/dev/ttyACM0", baud=57600):
    try:
        master = mavutil.mavlink_connection(port, baud=baud)
        print("📡 Pixhawk bağlantısı bekleniyor...")
        master.wait_heartbeat()
        print("✅ Pixhawk bağlantısı kuruldu.")
        return master
    except Exception as e:
        print(f"❌ Pixhawk bağlantı hatası: {e}")
        return None


# --- PWM komutu gönder ---
def pwm_gonder(master, kanal, pwm_deger):
    if master is None:
        return
    pwm_deger = max(min(pwm_deger, 1900), 1100)
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,
        kanal,
        pwm_deger,
        0, 0, 0, 0, 0
    )
    print(f"📤 Kanal {kanal} → PWM {pwm_deger}")


# --- Motorları durdur ---
def stop_motors(master):
    if master is None:
        return
    for i in range(1, 9):
        pwm_gonder(master, i, 1500)


# --- ARM komutu gönder ---
def arm_motors(master):
    if master is None:
        return
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1, 0, 0, 0, 0, 0, 0
    )
    print("✅ Motorlar ARM edildi.")


# Global değişkenler derinlik kontrolü için
Kp = 850.0
Ki = 65.0
Kd = 1000.0
hedef_derinlik = 0.6
integral = 0.0
previous_error = 0.0
last_time = time.time()
ilk_altitude = None
depth_control_active = True


def kontrol_motor_pid(master, derinlik):
    global integral, previous_error, last_time
    if master is None:
        return

    now = time.time()
    dt = now - last_time
    if dt <= 0.005:
        return
    last_time = now

    error = hedef_derinlik - derinlik
    integral += error * dt
    derivative = (error - previous_error) / dt
    previous_error = error

    output = Kp * error + Ki * integral + Kd * derivative

    if abs(output) < 100 and abs(error) > 0.01:
        output = 70 if output > 0 else -100

    pwm = int(1500 + output)
    pwm = max(min(pwm, 1900), 1100)

    print(f"📏 Göreli Derinlik: {derinlik:.3f} m | Hedef: {hedef_derinlik:.2f} m | PWM: {pwm}")

    for motor in [5, 6, 7, 8]:
        pwm_gonder(master, motor, pwm)


def depth_control_thread(master):
    global ilk_altitude, depth_control_active, shutdown_flag

    if master is None:
        return

    master.mav.command_long_send(
        target_system=1,
        target_component=1,
        command=mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        confirmation=0,
        param1=178,
        param2=200000,
        param3=0, param4=0, param5=0, param6=0, param7=0
    )
    print("📡 AHRS2 mesajı aktif edildi!")

    while depth_control_active and not shutdown_flag:
        try:
            msg = master.recv_match(type='AHRS2', blocking=True, timeout=1)
            if msg is None:
                continue

            altitude = msg.altitude
            if ilk_altitude is None:
                ilk_altitude = altitude
                print(f"📍 Başlangıç altitudesi: {ilk_altitude:.2f} m")
                continue

            derinlik = ilk_altitude - altitude
            kontrol_motor_pid(master, derinlik)
        except:
            continue


class LineFollowingAlgorithm:
    def _init_(self, video_path=None, model_path="best.pt"):
        self.cap = None
        self.width = 640  # Varsayılan değerler
        self.height = 480

        # 🆕 YOLO Model Yükleme - OPTIMIZED
        self.yolo_model = None
        self.model_path = model_path
        self.anomaly_count = 0
        self.last_anomaly_time = 0

        # 🔧 YOLO OPTIMİZASYONU - 20 FRAME INTERVAL
        self.yolo_frame_interval = 20  # 20 frame'de bir çalıştır
        self.yolo_frame_counter = 0
        self.last_anomaly_detections = []  # Son tespit edilen anomaliler
        self.anomaly_cooldown = {}  # Her anomali türü için cooldown
        self.anomaly_cooldown_duration = 60  # 60 frame (2 saniye) cooldown

        print(f"🔍 Model yolunu kontrol ediyor: {model_path}")
        print(f"🔍 YOLO kütüphanesi mevcut: {YOLO_AVAILABLE}")
        print(f"🔍 Model dosyası var mı: {os.path.exists(model_path)}")

        if YOLO_AVAILABLE and os.path.exists(model_path):
            try:
                print(f"🔄 Model yükleniyor: {model_path}")
                self.yolo_model = YOLO(model_path)
                print(f"✅ YOLO modeli başarıyla yüklendi! (Her {self.yolo_frame_interval} frame'de çalışacak)")

                # Model hakkında bilgi al
                if hasattr(self.yolo_model, 'names'):
                    print(f"🎯 Model sınıfları: {self.yolo_model.names}")
                else:
                    print(f"⚠ Model sınıf isimleri bulunamadı")

                # Anomali log dosyası oluştur
                self.anomaly_log_file = f"anomali_tespit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(self.anomaly_log_file, 'w', encoding='utf-8') as f:
                    f.write(f"ANOMALİ TESPİT RAPORU - OPTIMIZED\n")
                    f.write(f"Başlangıç Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"YOLO Interval: Her {self.yolo_frame_interval} frame\n")
                    f.write(f"Anomali Cooldown: {self.anomaly_cooldown_duration} frame\n")
                    f.write(f"Model: {model_path}\n")
                    if hasattr(self.yolo_model, 'names'):
                        f.write(f"Model Sınıfları: {self.yolo_model.names}\n")
                    f.write("=" * 60 + "\n\n")
                print(f"📝 Anomali log dosyası oluşturuldu: {self.anomaly_log_file}")

            except Exception as e:
                print(f"❌ YOLO model yükleme hatası: {e}")
                import traceback
                traceback.print_exc()
                self.yolo_model = None
        else:
            if not YOLO_AVAILABLE:
                print("❌ YOLO kütüphanesi yok - Çözüm: pip install ultralytics")
            elif not os.path.exists(model_path):
                print(f"❌ Model dosyası bulunamadı: {model_path}")
                print(f"🔍 Mevcut dizin: {os.getcwd()}")
                print(f"🔍 Dizin içeriği: {os.listdir('.')}")
            print("⚠ Sadece çizgi takibi aktif, anomali tespiti devre dışı")

        # Kamera optimizasyonu - DAHA HIZLI
        try:
            if video_path:
                self.cap = cv2.VideoCapture(video_path)
            else:
                # 🔧 KAMERA OPTIMİZASYONU
                self.cap = cv2.VideoCapture(0)
                # Daha düşük çözünürlük - daha yüksek FPS
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimum buffer
                # Otomatik ayarları kapat - daha stabil
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

            # Video boyutlarını al
            if self.cap.isOpened():
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"📷 Kamera başlatıldı: {self.width}x{self.height}")
            else:
                print("❌ Kamera başlatılamadı, varsayılan değerler kullanılıyor")
        except Exception as e:
            print(f"❌ Kamera başlatma hatası: {e}")

        # 🔧 İşleme boyutları - DAHA KÜÇÜK = DAHA HIZLI
        self.process_width = 240  # 320'den 240'a düşürdüm
        self.scale_factor = self.width / self.process_width if self.width > 0 else 1.0

        # Bölge ayarları
        self.region_width = self.process_width // 3
        self.region_height = int(self.height * 0.5 / self.scale_factor) if self.scale_factor > 0 else 120

        # Çizgi parametreleri
        self.lower_threshold = 0
        self.upper_threshold = 50
        self.max_allowed_angle = 20
        self.min_line_length = 30
        self.angle_correction_threshold = 15  # 20'den 15'e düşürdüm - daha hassas açı düzeltmesi

        # FPS
        self.prev_time = time.time()
        self.fps = 0

        # Arama parametreleri
        self.son_cizgi_yonu = "ORTA"
        self.cizgi_kayip_sayaci = 0
        self.kayip_esigi = 3
        self.minimum_pixel_esigi = 500

        # 🔧 K-means parametreleri - OPTIMİZE
        self.kmeans_clusters = 3
        self.use_kmeans = True
        self.kmeans_counter = 0
        self.kmeans_interval = 15  # 10'dan 15'e çıkardım - daha az sıklıkta çalışsın
        self.last_kmeans_mask = None

        # Subalti başlangıç parametreleri
        self.baslangic_suresi = 1.0
        self.baslangic_zamani = None
        self.algoritma_aktif = False

        # Master bağlantısı
        self.master = None

    def set_master(self, master):
        """Pixhawk bağlantısını ayarla"""
        self.master = master

    # 🔧 YOLO ANOMALİ TESPİTİ - OPTIMİZE EDİLDİ
    def detect_anomalies(self, frame):
        """YOLO ile anomali tespiti yap - OPTIMIZED"""
        anomalies = []

        if self.yolo_model is None:
            return anomalies, frame

        # 🔧 FRAME INTERVAL KONTROLÜ - 20 FRAME'DE BİR
        self.yolo_frame_counter += 1
        if self.yolo_frame_counter % self.yolo_frame_interval != 0:
            return anomalies, frame

        try:
            print(f"🔍 YOLO Frame {self.yolo_frame_counter} - Anomali tespiti çalışıyor...")

            # YOLO modelini çalıştır - optimize edilmiş ayarlar
            results = self.yolo_model(frame, verbose=False, conf=0.4, imgsz=320)  # Daha küçük imgsz

            current_time = time.time()
            current_frame = self.yolo_frame_counter

            for i, result in enumerate(results):
                boxes = result.boxes

                if boxes is not None:
                    for j, box in enumerate(boxes):
                        try:
                            # Tespit koordinatları
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0].cpu().numpy())
                            class_id = int(box.cls[0].cpu().numpy())

                            # Sınıf adını al
                            if hasattr(self.yolo_model, 'names') and class_id < len(self.yolo_model.names):
                                class_name = self.yolo_model.names[class_id]
                            else:
                                class_name = f"Anomali_{class_id}"

                            # 🔧 COOLDOWN KONTROLÜ - AYNI ANOMALİ TEKRAR TESPİT ETME
                            anomaly_key = f"{class_name}{int(x1 // 50)}{int(y1 // 50)}"  # Bölgesel gruplandırma

                            if anomaly_key in self.anomaly_cooldown:
                                if current_frame - self.anomaly_cooldown[anomaly_key] < self.anomaly_cooldown_duration:
                                    print(
                                        f"⏳ Cooldown: {class_name} ({self.anomaly_cooldown_duration - (current_frame - self.anomaly_cooldown[anomaly_key])} frame kaldı)")
                                    continue

                            # Yeni tespit veya cooldown bitti
                            self.anomaly_cooldown[anomaly_key] = current_frame

                            # Anomali bilgilerini sakla
                            anomaly_info = {
                                'class_name': class_name,
                                'confidence': confidence,
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'timestamp': datetime.now().strftime('%H:%M:%S.%f')[:-3],
                                'frame': current_frame
                            }
                            anomalies.append(anomaly_info)

                            print(f"🚨 YENİ ANOMALİ: {class_name} (Güven: {confidence:.3f}) Frame: {current_frame}")

                            # Frame üzerine çiz - BÜYÜK VE NET
                            color = (0, 0, 255)  # Kırmızı
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

                            # Etiket yaz
                            label = f"{class_name}: {confidence:.2f}"
                            font_scale = 0.8
                            thickness = 2
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

                            # Arka plan kutusu
                            cv2.rectangle(frame, (int(x1), int(y1) - label_size[1] - 15),
                                          (int(x1) + label_size[0] + 10, int(y1)), color, -1)

                            # Beyaz metin
                            cv2.putText(frame, label, (int(x1) + 5, int(y1) - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

                            # Log dosyasına yaz
                            self.log_anomaly(anomaly_info)
                            self.anomaly_count += 1
                            self.last_anomaly_time = current_time

                        except Exception as e:
                            print(f"❌ Kutu işleme hatası: {e}")

            if len(anomalies) > 0:
                print(f"🎯 TOPLAM {len(anomalies)} YENİ ANOMALİ TESPİT EDİLDİ!")

        except Exception as e:
            print(f"❌ YOLO anomali tespiti hatası: {e}")

        return anomalies, frame

    def log_anomaly(self, anomaly_info):
        """Anomali bilgilerini dosyaya yaz"""
        try:
            with open(self.anomaly_log_file, 'a', encoding='utf-8') as f:
                f.write(f"Zaman: {anomaly_info['timestamp']}\n")
                f.write(f"Frame: {anomaly_info['frame']}\n")
                f.write(f"Anomali Türü: {anomaly_info['class_name']}\n")
                f.write(f"Güven Oranı: {anomaly_info['confidence']:.3f}\n")
                f.write(f"Konum (x1,y1,x2,y2): {anomaly_info['bbox']}\n")
                f.write("-" * 40 + "\n")
                f.flush()
        except Exception as e:
            print(f"❌ Log yazma hatası: {e}")

    def cizgi_var_mi(self, regions):
        return any(region > self.minimum_pixel_esigi for region in regions)

    def son_cizgi_yonunu_guncelle(self, regions):
        sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions
        max_value = max(regions)
        max_index = regions.index(max_value)

        if max_index in [0, 3]:
            self.son_cizgi_yonu = "SOL"
        elif max_index in [1, 4]:
            self.son_cizgi_yonu = "ORTA"
        elif max_index in [2, 5]:
            self.son_cizgi_yonu = "SAG"

    def arama_modu_karar(self):
        if self.son_cizgi_yonu == "SOL":
            print("🔄 sol arama")
            pwm_gonder(self.master, 1, 1750)
            pwm_gonder(self.master, 2, 1530)
            pwm_gonder(self.master, 3, 1750)
            pwm_gonder(self.master, 4, 1530)
            return "SOL ARAMA"
        elif self.son_cizgi_yonu == "SAG":
            print("🔄 Sag arama")
            pwm_gonder(self.master, 1, 1570)
            pwm_gonder(self.master, 2, 1650)
            pwm_gonder(self.master, 3, 1570)
            pwm_gonder(self.master, 4, 1650)
            return "SAG ARAMA"
        else:
            print("🔄 Orta arama")
            pwm_gonder(self.master, 1, 1400)
            pwm_gonder(self.master, 2, 1400)
            pwm_gonder(self.master, 3, 1400)
            pwm_gonder(self.master, 4, 1400)
            return "ORTA ARAMA"

    # 🔧 PIKSel yoğunluğu hesaplama - OPTIMİZE
    def calculate_pixel_density(self, image):
        """Piksel yoğunluğunu hesapla - HIZLI"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

            # Daha küçük kernel - daha hızlı
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
            gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)

            return gradient_magnitude.astype(np.uint8)
        except:
            return np.zeros_like(image[:, :, 0] if len(image.shape) == 3 else image)

    # 🔧 HIZLI K-means - DAHA DA OPTIMİZE
    def fast_kmeans(self, features, k=3, max_iters=2):  # 3'ten 2'ye düşürdüm
        """Çok hızlı K-means implementasyonu"""
        try:
            n_samples, n_features = features.shape

            # Her 20. pikseli kullan (daha da hızlı)
            sample_indices = np.arange(0, n_samples, 20)
            sample_features = features[sample_indices]

            # Rastgele merkezler başlat
            np.random.seed(42)
            centroids = sample_features[np.random.choice(len(sample_features), k, replace=False)]

            for _ in range(max_iters):
                # Her noktayı en yakın merkeze ata
                distances = np.sqrt(((sample_features - centroids[:, np.newaxis]) ** 2).sum(axis=2))
                labels = np.argmin(distances, axis=0)

                # Merkezleri güncelle
                new_centroids = np.array([sample_features[labels == i].mean(axis=0) if np.sum(labels == i) > 0
                                          else centroids[i] for i in range(k)])

                centroids = new_centroids

            # Tüm noktalara etiket ata
            distances = np.sqrt(((features - centroids[:, np.newaxis]) ** 2).sum(axis=2))
            all_labels = np.argmin(distances, axis=0)

            return all_labels, centroids
        except:
            return np.random.randint(0, k, features.shape[0]), None

    # 🔧 HIZLI K-means segmentasyon - OPTIMİZE
    def segment_with_kmeans(self, image, pixel_density):
        """Çok hızlı K-means ile segmentasyon yap"""
        try:
            h, w = image.shape[:2]

            # Daha da küçük boyut - çok daha hızlı
            small_h, small_w = h // 3, w // 3  # 2'den 3'e çıkardım
            small_image = cv2.resize(image, (small_w, small_h))
            small_density = cv2.resize(pixel_density, (small_w, small_h))

            # Özellik vektörü oluştur
            features = []
            for y in range(small_h):
                for x in range(small_w):
                    if len(small_image.shape) == 3:
                        b, g, r = small_image[y, x]
                    else:
                        b = g = r = small_image[y, x]
                    density = small_density[y, x]
                    features.append([b, g, r, density])

            features = np.array(features, dtype=np.float32)

            # Hızlı K-means kümeleme
            labels, centroids = self.fast_kmeans(features, self.kmeans_clusters)

            # Sonucu küçük görüntü formatına çevir
            small_segmented = labels.reshape(small_h, small_w)

            # Orijinal boyuta geri büyüt
            segmented = cv2.resize(small_segmented.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)

            return segmented.astype(np.uint8), centroids
        except Exception as e:
            print(f"K-means hatası: {e}")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            _, thresh = cv2.threshold(gray, self.upper_threshold, 255, cv2.THRESH_BINARY_INV)
            return thresh // 255, None

    def identify_line_cluster(self, image, segmented, centroids):
        """Siyah çizgiye ait kümeyi belirle - SEÇİCİ"""
        try:
            if centroids is None:
                return 1

            cluster_stats = []
            h, w = segmented.shape

            for cluster_id in range(self.kmeans_clusters):
                mask = (segmented == cluster_id)

                if len(image.shape) == 3:
                    cluster_pixels = image[mask]
                else:
                    cluster_pixels = np.column_stack([image[mask]] * 3)

                if len(cluster_pixels) > 0:
                    mean_color = np.mean(cluster_pixels, axis=0)
                    mean_brightness = np.mean(mean_color)
                    pixel_count = np.sum(mask)

                    is_dark_enough = mean_brightness < 80
                    is_reasonable_size = 100 < pixel_count < (h * w * 0.3)

                    y_coords, x_coords = np.where(mask)
                    if len(y_coords) > 0:
                        width_span = np.max(x_coords) - np.min(x_coords)
                        height_span = np.max(y_coords) - np.min(y_coords)
                        aspect_ratio = max(width_span, height_span) / (min(width_span, height_span) + 1)
                        is_line_shaped = aspect_ratio > 3

                        center_x = np.mean(x_coords)
                        distance_from_center = abs(center_x - w / 2) / (w / 2)
                        is_center_aligned = distance_from_center < 0.7
                    else:
                        is_line_shaped = False
                        is_center_aligned = False

                    color_std = np.std(cluster_pixels, axis=0)
                    is_uniform = np.mean(color_std) < 30

                    line_score = 0
                    if is_dark_enough:
                        line_score += 40
                    if is_reasonable_size:
                        line_score += 30
                    if is_line_shaped:
                        line_score += 20
                    if is_center_aligned:
                        line_score += 10
                    if is_uniform:
                        line_score += 10

                    cluster_stats.append({
                        'id': cluster_id,
                        'score': line_score,
                        'brightness': mean_brightness,
                        'pixel_count': pixel_count
                    })

            valid_clusters = [c for c in cluster_stats if c['score'] >= 70]

            if valid_clusters:
                best_cluster = max(valid_clusters, key=lambda x: x['score'])
                return best_cluster['id']
            else:
                if cluster_stats:
                    darkest = min(cluster_stats, key=lambda x: x['brightness'])
                    return darkest['id']
                return 1

        except Exception as e:
            print(f"Çizgi tespit hatası: {e}")
            return 1

    def extract_kmeans_mask(self, segmented, line_cluster_id):
        """K-means sonucundan çizgi maskesini çıkar - OPTIMİZE"""
        try:
            line_mask = (segmented == line_cluster_id).astype(np.uint8) * 255

            # Küçük gürültüleri temizle - daha küçük kernel
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))  # 3'ten 2'ye
            line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, kernel_small)

            # Kontür analizi
            contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            clean_mask = np.zeros_like(line_mask)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 50 or area > line_mask.shape[0] * line_mask.shape[1] * 0.4:  # 100'den 50'ye
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / (min(w, h) + 1)

                if aspect_ratio > 2:
                    cv2.drawContours(clean_mask, [contour], -1, 255, -1)

            # Son temizlik - daha küçük kernel
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 5'ten 3'e
            clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)

            return clean_mask

        except Exception as e:
            print(f"Maske çıkarma hatası: {e}")
            return np.zeros(segmented.shape, dtype=np.uint8)

    # 🔧 PREPROCESS FRAME - OPTIMİZE
    def preprocess_frame(self, frame):
        try:
            # Daha küçük boyuta resize - daha hızlı işlem
            small_frame = cv2.resize(frame, (self.process_width, int(self.height / self.scale_factor)))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            # K-means kullanımı (daha az sıklıkta)
            if self.use_kmeans and self.kmeans_counter % self.kmeans_interval == 0:
                try:
                    pixel_density = self.calculate_pixel_density(small_frame)
                    segmented, centroids = self.segment_with_kmeans(small_frame, pixel_density)
                    line_cluster_id = self.identify_line_cluster(small_frame, segmented, centroids)
                    kmeans_mask = self.extract_kmeans_mask(segmented, line_cluster_id)

                    if np.sum(kmeans_mask) > 100 and np.sum(kmeans_mask) < (
                            small_frame.shape[0] * small_frame.shape[1] * 0.5):  # 200'den 100'e
                        self.last_kmeans_mask = kmeans_mask
                        self.kmeans_counter += 1
                        return gray, kmeans_mask, small_frame
                except Exception as e:
                    print(f"K-means işleminde hata: {e}")

            # Yakın zamanda K-means çalıştıysa onu kullan
            elif self.use_kmeans and self.last_kmeans_mask is not None and self.kmeans_counter % self.kmeans_interval < 8:  # 5'ten 8'e
                self.kmeans_counter += 1
                return gray, self.last_kmeans_mask, small_frame

            # Normal threshold
            _, thresh = cv2.threshold(gray, self.upper_threshold, 255, cv2.THRESH_BINARY_INV)
            kernel = np.ones((2, 2), np.uint8)  # 3'ten 2'ye küçült
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            self.kmeans_counter += 1
            return gray, thresh, small_frame
        except:
            empty = np.zeros((int(self.height / self.scale_factor), self.process_width), dtype=np.uint8)
            return empty, empty, empty

    def detect_line_angle(self, gray_frame):
        try:
            lines = cv2.HoughLinesP(gray_frame, 1, np.pi / 180,
                                    threshold=25,  # 30'dan 25'e düşürdüm
                                    minLineLength=self.min_line_length,
                                    maxLineGap=20)

            if lines is not None:
                lines = sorted(lines, key=lambda x: np.linalg.norm(x[0][2:] - x[0][:2]), reverse=True)[:2]
                angles = []
                centers = []
                valid_lines = []

                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                    if length > self.min_line_length:
                        angle = math.degrees(math.atan2(x2 - x1, y2 - y1))
                        if angle > 90:
                            angle -= 180
                        elif angle < -90:
                            angle += 180

                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        angles.append(angle)
                        centers.append(center)
                        valid_lines.append(line[0])

                if angles:
                    avg_angle = np.mean(angles)
                    avg_center = (int(np.mean([c[0] for c in centers])),
                                  int(np.mean([c[1] for c in centers])))
                    return avg_angle, avg_center, valid_lines
        except:
            pass
        return None, None, None

    def detect_line_position(self, thresh):
        try:
            upper_half = thresh[0:self.region_height, :]
            lower_half = thresh[self.region_height:, :]

            upper_regions = [
                np.count_nonzero(upper_half[:, i * self.region_width:(i + 1) * self.region_width])
                for i in range(3)
            ]

            lower_regions = [
                np.count_nonzero(lower_half[:, i * self.region_width:(i + 1) * self.region_width])
                for i in range(3)
            ]

            return upper_regions + lower_regions, thresh
        except:
            return [0, 0, 0, 0, 0, 0], thresh

    def is_line_angled(self, angle):
        if angle is None:
            return False
        return abs(angle) > self.angle_correction_threshold

    def viraj_tespiti(self, regions):
        sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions

        # 1. DÜZ ÇİZGİ KONTROLÜ (EN ÖNCE)
        if orta_alt > 1000 and orta_ust > 1000:
            return False

        # 2. YENGEÇ KONTROLÜ (SONRA)
        if sag_ust > 1000 and sag_alt > 1000:
            return False  # SAĞ YENGEÇ
        if sol_ust > 1000 and sol_alt > 1000:
            return False  # SOL YENGEÇ

        # 3. VİRAJ KONTROLÜ (EN SONRA)
        if sag_alt > 1000:
            return True  # SAĞ VİRAJ
        if sol_alt > 1000:
            return True  # SOL VİRAJ
        if orta_alt > 1000 and (sag_alt > 1000 or sol_alt > 1000):
            return True  # KARMA VİRAJ

        return False

    # 🔧 VİRAJ FONKSİYONU - FEEDBACK MEKANİZMASI EKLENDİ
    def viraj_fonksiyonu(self, regions):
        sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions

        # 🆕 FEEDBACK MEKANİZMASI - İSTEDİĞİN MANTIK
        # Orta altta pixel varsa, sol ve sağ alt karşılaştır
        if orta_alt > 1000:
            if sag_alt > sol_alt and sag_alt > 500:  # Sağ alt daha fazla
                print("🔄 saga don (feedback: sag_alt > sol_alt)")
                pwm_gonder(self.master, 1, 1490)
                pwm_gonder(self.master, 2, 1650)
                pwm_gonder(self.master, 3, 1490)
                pwm_gonder(self.master, 4, 1650)
                return "SAGA DON (FEEDBACK)"
            elif sol_alt > sag_alt and sol_alt > 500:  # Sol alt daha fazla
                print("🔄 sola don (feedback: sol_alt > sag_alt)")
                pwm_gonder(self.master, 1, 1900)
                pwm_gonder(self.master, 2, 1490)
                pwm_gonder(self.master, 3, 1900)
                pwm_gonder(self.master, 4, 1490)
                return "SOLA DON (FEEDBACK)"

        # ORİJİNAL VİRAJ MANTIGI (feedback olmadığında)
        # Sağ taraf virajları
        if sag_alt > 1000:
            if orta_alt > 1000:
                print("🔄 saga don (orta+sag)")
                pwm_gonder(self.master, 1, 1490)
                pwm_gonder(self.master, 2, 1650)
                pwm_gonder(self.master, 3, 1490)
                pwm_gonder(self.master, 4, 1650)
                return "SAGA DON"
            else:
                print("🔄 saga don (sadece sag)")
                pwm_gonder(self.master, 1, 1490)
                pwm_gonder(self.master, 2, 1650)
                pwm_gonder(self.master, 3, 1490)
                pwm_gonder(self.master, 4, 1650)
                return "SAGA DON"

        # Sol taraf virajları
        if sol_alt > 1000:
            if orta_alt > 1000:
                print("🔄 sola don (orta+sol)")
                pwm_gonder(self.master, 1, 1900)
                pwm_gonder(self.master, 2, 1490)
                pwm_gonder(self.master, 3, 1900)
                pwm_gonder(self.master, 4, 1490)
                return "SOLA DON"
            else:
                print("🔄 sola don (sadece sol)")
                pwm_gonder(self.master, 1, 1900)
                pwm_gonder(self.master, 2, 1490)
                pwm_gonder(self.master, 3, 1900)
                pwm_gonder(self.master, 4, 1490)
                return "SOLA DON"

        return "VIRAJ TESPIT EDILEMEDI"

    # 🔧 DÜZ ÇİZGİ FONKSİYONU - AÇI DÜZELTMESİ DÜZELTİLDİ
    def duz_cizgi_fonksiyonu(self, regions, angle=None, line_center=None):
        sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions

        # 1. ANA DÜZ ÇİZGİ KONTROLÜ (EN YÜKSEK ÖNCELİK)
        if orta_alt > 1000 and orta_ust > 1000:
            # 🔧 AÇI DÜZELTMESİ ŞARTINI DÜZELTTİM
            # Açı düzeltmesi yapılabilir koşulları:
            # a) Yan taraflarda çok güçlü çizgi yok
            # b) Açı belirsizliği var

            yan_cizgi_zayif = (sol_ust < 1500 and sag_ust < 1500 and sol_alt < 1500 and sag_alt < 1500)

            if yan_cizgi_zayif and self.is_line_angled(angle):
                if angle < -self.angle_correction_threshold:
                    print(f"🔄 saga don (açı düzeltmesi: {angle:.1f}°)")
                    pwm_gonder(self.master, 1, 1490)
                    pwm_gonder(self.master, 2, 1650)
                    pwm_gonder(self.master, 3, 1490)
                    pwm_gonder(self.master, 4, 1650)
                    return "SAGA DON (AÇI DÜZELTMESİ)"
                elif angle > self.angle_correction_threshold:
                    print(f"🔄 sola don (açı düzeltmesi: {angle:.1f}°)")
                    pwm_gonder(self.master, 1, 1700)
                    pwm_gonder(self.master, 2, 1490)
                    pwm_gonder(self.master, 3, 1700)
                    pwm_gonder(self.master, 4, 1490)
                    return "SOLA DON (AÇI DÜZELTMESİ)"

            print("🔄 duz git (orta güçlü)")
            pwm_gonder(self.master, 1, 1730)
            pwm_gonder(self.master, 2, 1700)
            pwm_gonder(self.master, 3, 1730)
            pwm_gonder(self.master, 4, 1700)
            return "DUZ GIT"

        # 2. YENGEÇ HAREKETLERİ (HEM ÜST HEM ALT)
        if sag_ust > 1000 and sag_alt > 1000:
            print("🔄 sag yengec (üst+alt)")
            pwm_gonder(self.master, 1, 1220)
            pwm_gonder(self.master, 2, 1750)
            pwm_gonder(self.master, 3, 1780)
            pwm_gonder(self.master, 4, 1250)
            return "SAG YENGEC"
        if sol_ust > 1000 and sol_alt > 1000:
            print("🔄 sol yengec (üst+alt)")
            pwm_gonder(self.master, 1, 1780)
            pwm_gonder(self.master, 2, 1250)
            pwm_gonder(self.master, 3, 1220)
            pwm_gonder(self.master, 4, 1750)
            return "SOL YENGEC"

        # 3. SADECE ÜST BÖLGE YENGEÇLERİ - AÇI DÜZELTMESİ EKLENDİ
        if sag_ust > 1000 and orta_ust <= 1000 and sol_ust <= 1000:
            # 🔧 AÇI DÜZELTMESİ YENGEC İÇİN DE EKLENDİ
            if self.is_line_angled(angle):
                if angle < -self.angle_correction_threshold:
                    print(f"🔄 saga don (sag yengec + açı düzeltmesi: {angle:.1f}°)")
                    pwm_gonder(self.master, 1, 1400)  # Daha agresif açı düzeltmesi
                    pwm_gonder(self.master, 2, 1750)
                    pwm_gonder(self.master, 3, 1780)
                    pwm_gonder(self.master, 4, 1400)
                    return "SAG YENGEC + AÇI DÜZELTMESİ"
                elif angle > self.angle_correction_threshold:
                    print(f"🔄 sola don (sag yengec ama açı solda: {angle:.1f}°)")
                    pwm_gonder(self.master, 1, 1650)
                    pwm_gonder(self.master, 2, 1400)
                    pwm_gonder(self.master, 3, 1400)
                    pwm_gonder(self.master, 4, 1650)
                    return "SAG YENGEC + SOLA AÇI DÜZELTMESİ"

            print("🔄 sag yengec (sadece üst)")
            pwm_gonder(self.master, 1, 1220)
            pwm_gonder(self.master, 2, 1750)
            pwm_gonder(self.master, 3, 1780)
            pwm_gonder(self.master, 4, 1250)
            return "SAG YENGEC"

        if sol_ust > 1000 and orta_ust <= 1000 and sag_ust <= 1000:
            # 🔧 AÇI DÜZELTMESİ SOL YENGEC İÇİN DE EKLENDİ
            if self.is_line_angled(angle):
                if angle > self.angle_correction_threshold:
                    print(f"🔄 sola don (sol yengec + açı düzeltmesi: {angle:.1f}°)")
                    pwm_gonder(self.master, 1, 1780)
                    pwm_gonder(self.master, 2, 1400)  # Daha agresif açı düzeltmesi
                    pwm_gonder(self.master, 3, 1400)
                    pwm_gonder(self.master, 4, 1750)
                    return "SOL YENGEC + AÇI DÜZELTMESİ"
                elif angle < -self.angle_correction_threshold:
                    print(f"🔄 saga don (sol yengec ama açı sağda: {angle:.1f}°)")
                    pwm_gonder(self.master, 1, 1400)
                    pwm_gonder(self.master, 2, 1650)
                    pwm_gonder(self.master, 3, 1650)
                    pwm_gonder(self.master, 4, 1400)
                    return "SOL YENGEC + SAGA AÇI DÜZELTMESİ"

            print("🔄 sol yengec (sadece üst)")
            pwm_gonder(self.master, 1, 1780)
            pwm_gonder(self.master, 2, 1250)
            pwm_gonder(self.master, 3, 1220)
            pwm_gonder(self.master, 4, 1750)
            return "SOL YENGEC"

        # 4. SADECE ORTA ÜST - GELİŞTİRİLMİŞ AÇI DÜZELTMESİ
        if orta_ust > 1000 and sol_ust <= 1000 and sag_ust <= 1000:
            if self.is_line_angled(angle):
                if angle < -self.angle_correction_threshold:
                    print(f"🔄 saga don (açı düzeltmesi - sadece orta üst: {angle:.1f}°)")
                    pwm_gonder(self.master, 1, 1470)
                    pwm_gonder(self.master, 2, 1650)
                    pwm_gonder(self.master, 3, 1470)
                    pwm_gonder(self.master, 4, 1650)
                    return "SAGA DON (AÇI DÜZELTMESİ)"
                elif angle > self.angle_correction_threshold:
                    print(f"🔄 sola don (açı düzeltmesi - sadece orta üst: {angle:.1f}°)")
                    pwm_gonder(self.master, 1, 1650)
                    pwm_gonder(self.master, 2, 1470)
                    pwm_gonder(self.master, 3, 1650)
                    pwm_gonder(self.master, 4, 1470)
                    return "SOLA DON (AÇI DÜZELTMESİ)"
            print("🔄 duz git (sadece orta üst)")
            pwm_gonder(self.master, 1, 1730)
            pwm_gonder(self.master, 2, 1700)
            pwm_gonder(self.master, 3, 1730)
            pwm_gonder(self.master, 4, 1700)
            return "DUZ GIT"

        # 5. SADECE ORTA ALT
        if orta_alt > 1000:
            print("🔄 geri git (sadece orta alt)")
            pwm_gonder(self.master, 1, 1400)
            pwm_gonder(self.master, 2, 1400)
            pwm_gonder(self.master, 3, 1400)
            pwm_gonder(self.master, 4, 1400)
            return "GERİ GİT"

        # 6. DİĞER DURUMLAR
        return "DUZ ÇİZGİ BELİRSİZ"

    # 🔧 DRAW INFO - OPTIMIZE EDİLDİ
    def draw_info(self, frame, mod, hareket, angle, regions, cizgi_mevcut, anomalies=None):
        """Bilgileri çiz - OPTIMIZE"""
        try:
            # Başlangıç durumu göstergesi
            if not self.algoritma_aktif:
                gecen_sure = time.time() - self.baslangic_zamani if self.baslangic_zamani else 0
                kalan_sure = max(0, self.baslangic_suresi - gecen_sure)
                cv2.putText(frame, f"BASLANGIC MODU - KALAN: {kalan_sure:.1f}s",
                            (10, self.height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.rectangle(frame, (2, 2), (self.width - 2, self.height - 2), (0, 255, 255), 4)

            # Temel bilgiler - daha kompakt
            info_y = 25
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (self.width - 100, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"Mod: {mod}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            info_y += 25
            cv2.putText(frame, f"Hareket: {hareket}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            info_y += 25
            aci_bilgisi = f"Açı: {angle:.1f}°" if angle is not None else "Açı: --"
            cv2.putText(frame, aci_bilgisi, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            info_y += 25
            cv2.putText(frame, f"Cizgi: {'VAR' if cizgi_mevcut else 'YOK'}", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if cizgi_mevcut else (0, 0, 255), 1)

            # 🔧 K-means ve YOLO durumu - OPTIMIZE
            info_y += 25
            kmeans_status = f"K-means: {self.kmeans_counter % self.kmeans_interval}/{self.kmeans_interval}" if self.use_kmeans else "K-means: OFF"
            cv2.putText(frame, kmeans_status, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 0) if self.use_kmeans else (0, 0, 255), 1)

            # YOLO bilgisi - OPTIMIZE
            info_y += 20
            if self.yolo_model is not None:
                yolo_status = f"YOLO: {self.yolo_frame_counter % self.yolo_frame_interval}/{self.yolo_frame_interval} | {self.anomaly_count} total"
                cv2.putText(frame, yolo_status, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)

                # Aktif anomaliler
                if anomalies and len(anomalies) > 0:
                    info_y += 20
                    cv2.putText(frame, f"AKTIF: {len(anomalies)} ANOMALİ!", (10, info_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Bölgeler - daha kompakt
            info_y += 25
            sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions
            region_text = f"U[{sol_ust},{orta_ust},{sag_ust}] A[{sol_alt},{orta_alt},{sag_alt}]"
            cv2.putText(frame, region_text, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Son yön bilgisi
            info_y += 20
            cv2.putText(frame, f"Son Yön: {self.son_cizgi_yonu} | Kayıp: {self.cizgi_kayip_sayaci}",
                        (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            # Bölge çizgileri - daha ince
            rw = int(self.region_width * self.scale_factor)
            rh = int(self.region_height * self.scale_factor)
            cv2.line(frame, (rw, 0), (rw, self.height), (0, 255, 0), 1)
            cv2.line(frame, (rw * 2, 0), (rw * 2, self.height), (0, 255, 0), 1)
            cv2.line(frame, (0, rh), (self.width, rh), (0, 255, 0), 1)

            # Merkez çizgisi
            cv2.line(frame, (self.width // 2, 0), (self.width // 2, self.height), (255, 0, 0), 1)

            # Mod çerçevesi
            if mod == "ARAMA MODU":
                cv2.rectangle(frame, (2, 2), (self.width - 2, self.height - 2), (0, 0, 255), 3)
            elif "VIRAJ" in mod:
                cv2.rectangle(frame, (2, 2), (self.width - 2, self.height - 2), (255, 0, 0), 3)
            elif "YENGEC" in hareket:
                cv2.rectangle(frame, (2, 2), (self.width - 2, self.height - 2), (0, 255, 255), 3)
            elif anomalies and len(anomalies) > 0:
                cv2.rectangle(frame, (2, 2), (self.width - 2, self.height - 2), (255, 0, 255), 4)

        except:
            pass

    # 🔧 ANA DÖNGÜ - OPTIMİZE EDİLDİ
    def run(self):
        """Ana döngü - ULTRA OPTIMIZE"""
        print("*** ÇİZGİ TAKİBİ + YOLO ANOMALİ TESPİTİ - ULTRA OPTIMIZE ***")
        print("🚀 PERFORMANCE OPTIMIZATIONS:")
        print(f"   📷 İşleme boyutu: {self.process_width}px (küçültüldü)")
        print(f"   🧠 K-means interval: {self.kmeans_interval} frame")
        print(f"   🎯 YOLO interval: {self.yolo_frame_interval} frame")
        print(f"   ⚡ Açı düzeltme eşiği: {self.angle_correction_threshold}°")
        print("🌊 İLK 1 SANİYE DÜZ İLERİ GİDECEK, SONRA ALGORİTMA ÇALIŞACAK")

        # YOLO durumu
        if self.yolo_model:
            print(f"✅ YOLO AKTİF - Her {self.yolo_frame_interval} frame'de çalışacak")
            print(f"📝 Log dosyası: {self.anomaly_log_file}")
            if hasattr(self.yolo_model, 'names'):
                print(f"🎯 Tespit sınıfları: {list(self.yolo_model.names.values())}")
        else:
            print("❌ YOLO modeli yok - Sadece çizgi takibi")

        print("Çıkmak için 'q', Duraklatmak için 'SPACE', Sıfırlamak için 'r'")

        # Kamera kontrolü
        if self.cap is None or not self.cap.isOpened():
            print("❌ Kamera başlatılamadı!")
            return

        paused = False
        frame_count = 0
        last_fps_time = time.time()

        # Başlangıç zamanı
        self.baslangic_zamani = time.time()
        print(f"🚀 BAŞLANGIC: {self.baslangic_suresi} saniye düz gidecek...")

        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("Video bitti!")
                    break

                frame_count += 1
                current_time = time.time()

                # FPS hesapla (her 30 frame'de)
                if frame_count % 30 == 0:
                    if current_time - last_fps_time > 0:
                        self.fps = 30 / (current_time - last_fps_time)
                    last_fps_time = current_time

                # BAŞLANGIÇ KONTROL MANTIGI
                if not self.algoritma_aktif:
                    gecen_sure = current_time - self.baslangic_zamani

                    if gecen_sure >= self.baslangic_suresi:
                        self.algoritma_aktif = True
                        print(f"✅ BAŞLANGIÇ TAMAMLANDI! Algoritma aktif. ({gecen_sure:.1f}s)")
                    else:
                        # Başlangıç modu - sadece düz git
                        kalan_sure = self.baslangic_suresi - gecen_sure
                        hareket = "DUZ GIT (BAŞLANGIÇ)"
                        mod = "BAŞLANGIÇ MODU"

                        # Başlangıçta da YOLO çalışabilir (daha az sıklıkta)
                        anomalies = []
                        if self.yolo_model is not None:
                            anomalies, frame = self.detect_anomalies(frame)
                            if anomalies:
                                mod += " + ANOMALİ"

                        angle = None
                        regions = [0, 0, 0, 0, 0, 0]
                        cizgi_mevcut = False

                        # Görselleştirme
                        self.draw_info(frame, mod, hareket, angle, regions, cizgi_mevcut, anomalies)
                        cv2.imshow('Cizgi Takibi + YOLO Anomali', frame)

                        # Konsol çıktısı (her 30 frame'de)
                        if frame_count % 30 == 0:
                            anomali_info = f" | {len(anomalies)} anomali" if anomalies else ""
                            print(f"🌊 BAŞLANGIÇ: Kalan {kalan_sure:.1f}s{anomali_info}")

                        # Tuş kontrol ve devam
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord(' '):
                            paused = not paused
                        elif key == ord('a'):
                            self.algoritma_aktif = True
                            print("🚀 ALGORİTMA MANUEL BAŞLATILDI!")
                        elif key == ord('r'):
                            self.cizgi_kayip_sayaci = 0
                            self.son_cizgi_yonu = "ORTA"
                            self.algoritma_aktif = False
                            self.baslangic_zamani = time.time()
                            print("🔄 Sistem sıfırlandı")
                        continue

                # ALGORİTMA AKTİF - ANA İŞLEME
                # YOLO anomali tespiti (optimize edilmiş interval)
                anomalies = []
                if self.yolo_model is not None:
                    anomalies, frame = self.detect_anomalies(frame)

                # Görüntü işleme (optimize edilmiş)
                gray, thresh, small_frame = self.preprocess_frame(frame)
                angle, center, lines = self.detect_line_angle(thresh)
                regions, _ = self.detect_line_position(thresh)

                # Çizgi kontrolü
                cizgi_mevcut = self.cizgi_var_mi(regions)

                # Hareket karar algoritması
                if cizgi_mevcut:
                    self.cizgi_kayip_sayaci = 0
                    self.son_cizgi_yonunu_guncelle(regions)

                    is_viraj = self.viraj_tespiti(regions)

                    if is_viraj:
                        hareket = self.viraj_fonksiyonu(regions)  # Feedback mekanizmalı
                        mod = "VIRAJ MODU"
                    else:
                        hareket = self.duz_cizgi_fonksiyonu(regions, angle, center)  # Düzeltilmiş açı kontrolü
                        mod = "DUZ CIZGI MODU"
                else:
                    self.cizgi_kayip_sayaci += 1

                    if self.cizgi_kayip_sayaci >= self.kayip_esigi:
                        hareket = self.arama_modu_karar()
                        mod = "ARAMA MODU"
                    else:
                        hareket = "BEKLE"
                        mod = "BEKLE MODU"

                # Anomali durumunda mod güncelle
                if anomalies:
                    mod += " + ANOMALİ"

                # Görselleştirme
                self.draw_info(frame, mod, hareket, angle, regions, cizgi_mevcut, anomalies)

                # Tespit edilen çizgileri çiz
                if lines:
                    for line in lines:
                        x1, y1, x2, y2 = [int(x * self.scale_factor) for x in line]
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Görüntüleri göster
                cv2.imshow('Cizgi Takibi + YOLO Anomali', frame)

                # Threshold küçük boyutta (performans için)
                if frame_count % 5 == 0 and thresh is not None:  # Her 5 frame'de bir göster
                    small_thresh = cv2.resize(thresh, (240, 180))  # Daha küçük
                    cv2.imshow('Threshold', small_thresh)

                # Konsol çıktısı (daha az sıklıkta)
                if frame_count % 90 == 0:  # 60'tan 90'a çıkardım
                    print(f"📊 Frame: {frame_count}, FPS: {self.fps:.1f}")
                    print(f"    Mod: {mod}, Hareket: {hareket}")
                    if self.yolo_model:
                        active_anomalies = len(anomalies) if anomalies else 0
                        print(f"    YOLO: {active_anomalies} aktif, {self.anomaly_count} toplam tespit")

            # Tuş kontrolü
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Çıkış")
                break
            elif key == ord(' '):
                paused = not paused
                print("⏸ Duraklatıldı" if paused else "▶ Devam")
            elif key == ord('r'):
                self.cizgi_kayip_sayaci = 0
                self.son_cizgi_yonu = "ORTA"
                self.algoritma_aktif = False
                self.baslangic_zamani = time.time()
                print("🔄 Sistem sıfırlandı")
            elif key == ord('k'):
                self.use_kmeans = not self.use_kmeans
                print(f"🔧 K-means {'AKTİF' if self.use_kmeans else 'PASİF'}")
            elif key == ord('i'):
                self.kmeans_interval = 10 if self.kmeans_interval == 15 else 15
                print(f"🔧 K-means interval: {self.kmeans_interval}")
            elif key == ord('y'):  # YOLO interval ayarı
                self.yolo_frame_interval = 10 if self.yolo_frame_interval == 20 else 20
                print(f"🔧 YOLO interval: {self.yolo_frame_interval}")
            elif key == ord('a'):
                if not self.algoritma_aktif:
                    self.algoritma_aktif = True
                    print("🚀 ALGORİTMA MANUEL BAŞLATILDI!")

        # Temizlik
        self.cap.release()
        cv2.destroyAllWindows()

        # Anomali log tamamla
        if self.yolo_model and self.anomaly_count > 0:
            try:
                with open(self.anomaly_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n" + "=" * 60 + "\n")
                    f.write(f"TOPLAM TESPİT: {self.anomaly_count}\n")
                    f.write(f"PROGRAM BİTİŞ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"FPS PERFORMANSI: {self.fps:.1f}\n")
                print(f"📝 Log tamamlandı: {self.anomaly_count} tespit, FPS: {self.fps:.1f}")
                print(f"📁 Log: {self.anomaly_log_file}")
            except:
                pass

    def cleanup(self):
        """Temizlik fonksiyonu"""
        if self.cap is not None:
            self.cap.release()


def main():
    """Ana fonksiyon"""
    global shutdown_flag, depth_control_active

    algorithm = None
    master = None

    try:
        print("🔧 Jetson Nano sistemi başlatılıyor...")

        # Pixhawk bağlantısı
        print("🚁 Pixhawk bağlantısı deneniyor...")
        master = baglanti_kur()

        if master:
            arm_motors(master)
            time.sleep(1)

            # Derinlik kontrol başlat
            depth_thread = threading.Thread(target=depth_control_thread, args=(master,))
            depth_thread.daemon = True
            depth_thread.start()
            print("🌊 Derinlik kontrol başlatıldı")
        else:
            print("⚠ Pixhawk bağlanamadı, sadece kamera modu")

        # Model yolu belirleme
        possible_paths = [
            "best.pt",
            "./best.pt",
            "/home/kubra/modellıdenenecekler/best.pt",
            "/home/kubra/İndirilenler/best.pt",
            "models/best.pt"
        ]

        model_path = "best.pt"  # Varsayılan
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f"✅ Model bulundu: {path}")
                break

        # Algoritma başlat
        print("🚀 ULTRA OPTIMIZE algoritma başlatılıyor...")
        algorithm = LineFollowingAlgorithm(model_path=model_path)
        algorithm.set_master(master)

        if shutdown_flag:
            print("🛑 Başlatma iptal")
            return

        algorithm.run()

    except KeyboardInterrupt:
        print("\n🛑 Program durduruldu")
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
    finally:
        shutdown_flag = True
        depth_control_active = False

        print("🛑 Sistem kapatılıyor...")

        if master:
            try:
                stop_motors(master)
                print("🔴 Motorlar durduruldu")
            except Exception as e:
                print(f"Motor durdurma hatası: {e}")

        if algorithm:
            try:
                algorithm.cleanup()
            except Exception as e:
                print(f"Temizlik hatası: {e}")

        print("✅ Program güvenli şekilde sonlandı")


if _name_ == "_main_":
    main()