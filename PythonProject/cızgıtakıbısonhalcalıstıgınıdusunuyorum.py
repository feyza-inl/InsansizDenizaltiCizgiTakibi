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
import json

# IMU import
try:
    from Adafruit_BNO055 import BNO055

    IMU_AVAILABLE = True
    print("✅ IMU kütüphanesi yüklendi")
except ImportError:
    IMU_AVAILABLE = False
    print("⚠ IMU kütüphanesi bulunamadı")

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

# ==========================
# IMU ve DERINLIK KONTROL AYARLARI
# ==========================
SERIAL_IMU = '/dev/ttyUSB1'
CONFIG_FILE = 'referans_nokta.json'

# Hedef değerler
HEDEF_DERINLIK = 0.6  # metre
HEDEF_PITCH = 0.0  # derece (0 = yatay)

# PID ayarları - Derinlik
Kp_derinlik = 860.0
Ki_derinlik = 65.0
Kd_derinlik = 1000.0

# PID ayarları - Pitch
Kp_pitch = 5.0
Ki_pitch = 0.5
Kd_pitch = 3.0

# PWM değerleri
PWM_NOTR = 1500
PWM_MIN = 1100
PWM_MAX = 1900

# Motor konfigürasyonu (Vertical thrusters)
MOTOR_ON_SAG = 5
MOTOR_ON_SOL = 6
MOTOR_ARKA_SAG = 7
MOTOR_ARKA_SOL = 8

# Global değişkenler IMU ve derinlik kontrolü için
imu = None
ilk_altitude = None
integral_derinlik = 0.0
previous_error_derinlik = 0.0
last_time_derinlik = time.time()
integral_pitch = 0.0
previous_error_pitch = 0.0
last_time_pitch = time.time()
depth_control_active = True


# ==========================
# YARDIMCI FONKSİYONLAR
# ==========================
def clamp(val, lo, hi):
    return max(min(val, hi), lo)


def aci_norm(d):
    return d % 360


def heading_fark(hedef, mevcut):
    fark = hedef - mevcut
    if fark > 180:
        fark -= 360
    elif fark < -180:
        fark += 360
    return fark


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
    pwm_deger = clamp(pwm_deger, PWM_MIN, PWM_MAX)
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
    for i in [MOTOR_ON_SAG, MOTOR_ON_SOL, MOTOR_ARKA_SAG, MOTOR_ARKA_SOL]:
        pwm_gonder(master, i, PWM_NOTR)


def tum_motorlari_durdur(master):
    """Tüm motorları durdur"""
    stop_motors(master)


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


# ==========================
# IMU FONKSİYONLARI
# ==========================
def imu_aci_oku():
    """IMU'dan pitch açısını oku"""
    global imu
    if imu is None:
        return None
    try:
        _, p, _ = imu.read_euler()  # pitch değerini al
        if p is not None:
            return p
        return None
    except:
        return None


def guvenli_pitch_oku():
    """Güvenli pitch okuma - birkaç deneme yap"""
    for i in range(3):  # Daha az deneme - performans için
        pitch = imu_aci_oku()
        if pitch is not None:
            return pitch
        time.sleep(0.05)
    return None


def referans_kaydet():
    """Referans pitch açısını kaydet"""
    global imu
    print("\n" + "=" * 60)
    print("🎯 REFERANS NOKTA BELİRLEME")
    print("=" * 60)

    print("\n📍 Referans nokta kaydediliyor...")
    okumalar = []
    for i in range(10):  # 20'den 10'a düşürdüm
        aci = guvenli_pitch_oku()
        if aci is not None:
            okumalar.append(aci)
        print(f"Okuma {i + 1}/10: {aci}°" if aci else f"Okuma {i + 1}/10: HATA")
        time.sleep(0.2)

    if len(okumalar) < 5:
        raise RuntimeError("❌ Yeterli okuma yapılamadı!")

    en_iyi_aci = okumalar[0]
    en_iyi_sayi = 0

    for test_aci in okumalar:
        sayi = sum(1 for oku_aci in okumalar if abs(test_aci - oku_aci) < 5)
        if sayi > en_iyi_sayi:
            en_iyi_sayi = sayi
            en_iyi_aci = test_aci

    with open(CONFIG_FILE, 'w') as f:
        json.dump({
            'referans_pitch': en_iyi_aci,
            'zaman': time.time(),
            'okuma_sayisi': len(okumalar),
            'tutarlilik': en_iyi_sayi
        }, f, indent=2)

    print(f"\n✅ REFERANS PITCH KAYDEDİLDİ!")
    print(f"📐 Referans pitch: {en_iyi_aci:.1f}°")
    return en_iyi_aci


def referans_oku():
    """Kayıtlı referans pitch açısını oku"""
    if not os.path.exists(CONFIG_FILE):
        return None
    try:
        with open(CONFIG_FILE, 'r') as f:
            data = json.load(f)
            return data.get('referans_pitch')
    except:
        return None


# ==========================
# DERINLIK ve PITCH KONTROLÜ
# ==========================
def derinlik_oku(master):
    """Pixhawk'tan altitude verisi oku"""
    msg = master.recv_match(type='AHRS2', blocking=False)
    if msg is not None:
        return msg.altitude
    return None


def derinlik_kontrol(master):
    """Derinlik kontrolü"""
    global ilk_altitude, integral_derinlik, previous_error_derinlik, last_time_derinlik

    # İlk altitude kaydı
    if ilk_altitude is None:
        altitude = derinlik_oku(master)
        if altitude is not None:
            ilk_altitude = altitude
            print(f"📍 Başlangıç altitudesi referans alındı: {ilk_altitude:.2f} m")
        return None, PWM_NOTR

    # Derinlik okuma
    altitude = derinlik_oku(master)
    if altitude is None:
        return None, PWM_NOTR

    # Göreli derinlik: ne kadar aşağı indik?
    derinlik = ilk_altitude - altitude

    # PID kontrol
    now = time.time()
    dt = now - last_time_derinlik
    if dt <= 0.005:
        return derinlik, PWM_NOTR
    last_time_derinlik = now

    error = HEDEF_DERINLIK - derinlik
    integral_derinlik += error * dt
    derivative = (error - previous_error_derinlik) / dt
    previous_error_derinlik = error

    output = Kp_derinlik * error + Ki_derinlik * integral_derinlik + Kd_derinlik * derivative

    # Ufak hatalarda küçük kuvvet uygula
    if abs(output) < 100 and abs(error) > 0.01:
        output = 70 if output > 0 else -100

    pwm_derinlik = int(PWM_NOTR + output)
    pwm_derinlik = clamp(pwm_derinlik, PWM_MIN, PWM_MAX)

    return derinlik, pwm_derinlik


def pitch_kontrol():
    """Pitch açısı kontrolü"""
    global integral_pitch, previous_error_pitch, last_time_pitch

    # Pitch okuma
    pitch = guvenli_pitch_oku()
    if pitch is None:
        return None, 0

    # PID kontrol
    now = time.time()
    dt = now - last_time_pitch
    if dt <= 0.005:
        return pitch, 0
    last_time_pitch = now

    error = HEDEF_PITCH - pitch
    integral_pitch += error * dt
    integral_pitch = clamp(integral_pitch, -200, 200)
    derivative = (error - previous_error_pitch) / dt
    previous_error_pitch = error

    output = Kp_pitch * error + Ki_pitch * integral_pitch + Kd_pitch * derivative
    pwm_pitch = int(output * 1.5)  # Ölçeklendirme

    return pitch, pwm_pitch


def kombine_kontrol(master):
    """Derinlik ve pitch kontrolünü kombine et"""
    if master is None:
        return

    # Derinlik kontrolü
    derinlik, pwm_derinlik = derinlik_kontrol(master)

    # Pitch kontrolü
    pitch, pwm_pitch = pitch_kontrol()

    if derinlik is None or pitch is None:
        return

    # Motorlara PWM değerlerini uygula
    # Ön motorlar: derinlik - pitch (pitch düzeltmesi için ters)
    # Arka motorlar: derinlik + pitch
    pwm_on_sag = clamp(pwm_derinlik - pwm_pitch, PWM_MIN, PWM_MAX)
    pwm_on_sol = clamp(pwm_derinlik - pwm_pitch, PWM_MIN, PWM_MAX)
    pwm_arka_sag = clamp(pwm_derinlik + pwm_pitch, PWM_MIN, PWM_MAX)
    pwm_arka_sol = clamp(pwm_derinlik + pwm_pitch, PWM_MIN, PWM_MAX)

    pwm_gonder(master, MOTOR_ON_SAG, pwm_on_sag)
    pwm_gonder(master, MOTOR_ON_SOL, pwm_on_sol)
    pwm_gonder(master, MOTOR_ARKA_SAG, pwm_arka_sag)
    pwm_gonder(master, MOTOR_ARKA_SOL, pwm_arka_sol)

    print(f"📏 Derinlik: {derinlik:.3f}m | 🎯 Pitch: {pitch:.1f}° | "
          f"⚙ Ön: {pwm_on_sag} | Arka: {pwm_arka_sag}")


def depth_control_thread(master):
    """Derinlik ve pitch kontrol thread'i"""
    global depth_control_active, shutdown_flag

    if master is None:
        return

    # AHRS2 mesajını aktif et
    master.mav.command_long_send(
        target_system=1,
        target_component=1,
        command=mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        confirmation=0,
        param1=178,  # AHRS2 ID
        param2=200000,  # 200ms = 5Hz
        param3=0, param4=0, param5=0, param6=0, param7=0
    )
    print("📡 AHRS2 mesajı aktif edildi!")

    while depth_control_active and not shutdown_flag:
        try:
            kombine_kontrol(master)
            time.sleep(0.05)  # 20Hz kontrol döngüsü
        except Exception as e:
            print(f"Derinlik kontrol hatası: {e}")
            time.sleep(0.1)


class LineFollowingAlgorithm:
    def __init__(self, video_path=None, model_path="best.pt"):
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
        else:
            # Kamera ayarlarını optimize et - BİRİNCİ KODDAKI AYARLAR
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Video boyutlarını al
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # İşleme için küçültülmüş boyutlar - BİRİNCİ KODDAKI AYARLAR
        self.process_width = 320
        self.scale_factor = self.width / self.process_width

        # Ayarlanabilir bölge genişlikleri - BİRİNCİ KODDAKI SUPERIOR BÖLGE AYARLARI
        self.region_ratios = [0.30, 0.40, 0.30]  # Sol, Orta, Sağ oranları
        self.region_widths = [
            int(self.process_width * self.region_ratios[0]),  # Sol genişlik
            int(self.process_width * self.region_ratios[1]),  # Orta genişlik
            int(self.process_width * self.region_ratios[2])  # Sağ genişlik
        ]
        self.region_height = int(self.height * 0.5 / self.scale_factor)

        # Çizgi rengi eşiği (siyah çizgi için) - BİRİNCİ KODDAKI AYARLAR
        self.lower_threshold = 0
        self.upper_threshold = 50

        # *** DÜZELTILMIŞ PARAMETRELER - VIRAJ ÖNCELIKLI SISTEM - BİRİNCİ KODDAN ***
        self.max_allowed_angle = 15
        self.min_line_length = 25
        self.angle_correction_threshold = 15  # Arttırıldı, daha az hassas

        # *** VIRAJ MODU KARARLILIĞI İÇİN YENİ PARAMETRELER - BİRİNCİ KODDAN ***
        self.viraj_modu_aktif = False
        self.viraj_modu_suresi = 0
        self.min_viraj_suresi = 10  # En az 8 frame viraj modunda kalsın
        self.viraj_cikis_esigi = 3  # 3 frame boyunca viraj tespit edilmezse çık

        # VİRAJ TESPİTİ PARAMETRELERİ - BİRİNCİ KODDAN
        self.orta_ust_viraj_killer = 500
        self.viraj_pixel_threshold = 1000
        self.viraj_dominance_ratio = 1.05

        # NORMAL İŞLEM PARAMETRELERİ - BİRİNCİ KODDAN
        self.normal_pixel_esigi = 800
        self.minimum_pixel_esigi = 600
        self.dominance_ratio = 1.3

        # FPS hesaplama
        self.prev_time = time.time()
        self.fps = 0

        # ARAMA MODU PARAMETRELERİ - BİRİNCİ KODDAN
        self.son_cizgi_yonu = "ORTA"
        self.cizgi_kayip_sayaci = 0
        self.kayip_esigi = 5

        # DURUM KONTROLÜ
        self.current_state = "NORMAL"
        self.state_counter = 0

        # YOLO Model Yükleme
        self.yolo_model = None
        self.model_path = model_path
        self.anomaly_count = 0
        self.last_anomaly_time = 0
        self.yolo_frame_interval = 30  # 20 frame'de bir çalıştır
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

        # Master bağlantısı
        self.master = None

    def set_master(self, master):
        """Pixhawk bağlantısını ayarla"""
        self.master = master

    # *** BİRİNCİ KODDAKI MÜKEMMEL FONKSİYONLAR - TAMAMEN KORUNACAK ***
    def cizgi_var_mi(self, regions):
        """Çizgi var mı yok mu kontrol et - BİRİNCİ KODDAN"""
        return any(region > self.minimum_pixel_esigi for region in regions)

    def son_cizgi_yonunu_guncelle(self, regions):
        """Son görülen çizginin yönünü güncelle - BİRİNCİ KODDAN"""
        sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions

        max_value = max(regions)
        max_index = regions.index(max_value)

        if max_index in [0, 3]:  # Sol üst veya sol alt
            self.son_cizgi_yonu = "SOL"
        elif max_index in [1, 4]:  # Orta üst veya orta alt
            self.son_cizgi_yonu = "ORTA"
        elif max_index in [2, 5]:  # Sağ üst veya sağ alt
            self.son_cizgi_yonu = "SAG"

    def arama_modu_karar(self):
        """Arama modunda hangi yöne gidileceğini belirle - BİRİNCİ KODDAN"""
        if self.son_cizgi_yonu == "SOL":
            return "SOL ARAMA"
        elif self.son_cizgi_yonu == "SAG":
            return "SAG ARAMA"
        else:
            return "ORTA ARAMA"

    def preprocess_frame(self, frame):
        """Görüntüyü küçült ve ön işleme yap - BİRİNCİ KODDAKI MÜKEMMEL YÖNTEM"""
        small_frame = cv2.resize(frame, (self.process_width, int(self.height / self.scale_factor)))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.upper_threshold, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        return gray, thresh, small_frame

    def detect_line_angle(self, gray_frame):
        """Optimize edilmiş çizgi açısı tespiti - BİRİNCİ KODDAKI MÜKEMMEL YÖNTEM"""
        lines = cv2.HoughLinesP(gray_frame, 1, np.pi / 180,
                                threshold=25,
                                minLineLength=self.min_line_length,
                                maxLineGap=15)

        if lines is not None and len(lines) > 0:
            lines = sorted(lines, key=lambda x: np.linalg.norm(x[0][2:] - x[0][:2]), reverse=True)[:3]

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

                    if abs(angle) < 45:
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        angles.append(angle)
                        centers.append(center)
                        valid_lines.append(line[0])

            if angles:
                avg_angle = np.mean(angles)
                avg_center = (int(np.mean([c[0] for c in centers])),
                              int(np.mean([c[1] for c in centers])))
                return avg_angle, avg_center, valid_lines

        return None, None, None

    def detect_line_position(self, thresh):
        """Optimize edilmiş bölge tespiti - BİRİNCİ KODDAKI MÜKEMMEL YÖNTEM"""
        upper_half = thresh[0:self.region_height, :]
        lower_half = thresh[self.region_height:, :]

        upper_regions = [
            np.count_nonzero(upper_half[:, 0:self.region_widths[0]]),  # Sol üst
            np.count_nonzero(upper_half[:, self.region_widths[0]:self.region_widths[0] + self.region_widths[1]]),
            # Orta üst
            np.count_nonzero(upper_half[:, self.region_widths[0] + self.region_widths[1]:])  # Sağ üst
        ]

        lower_regions = [
            np.count_nonzero(lower_half[:, 0:self.region_widths[0]]),  # Sol alt
            np.count_nonzero(lower_half[:, self.region_widths[0]:self.region_widths[0] + self.region_widths[1]]),
            # Orta alt
            np.count_nonzero(lower_half[:, self.region_widths[0] + self.region_widths[1]:])  # Sağ alt
        ]

        return upper_regions + lower_regions, thresh

    def is_line_angled(self, angle):
        """Çizginin açılı olup olmadığını kontrol eder - BİRİNCİ KODDAN"""
        if angle is None or self.viraj_modu_aktif:
            return False
        return abs(angle) > self.angle_correction_threshold

    def viraj_tespit_et(self, regions):
        """ÖNCELIKLI VIRAJ TESPIT FONKSIYONU - BİRİNCİ KODDAKI MÜKEMMEL YÖNTEM"""
        sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions

        print(f"VIRAJ DEBUG: orta_ust={orta_ust}, orta_alt={orta_alt}, sag_alt={sag_alt}, sol_alt={sol_alt}")

        # *** ÇAKIŞMA ÖNLEYİCİ - Orta üst çok yüksekse viraj değil ***
        if orta_ust > self.orta_ust_viraj_killer:
            print(f"VIRAJ DEBUG: Orta üst çok yüksek ({orta_ust}) - Viraj değil")
            return False

        # *** GÜÇLÜ VİRAJ TESPİTLERİ ***

        # SAĞ VİRAJ: Sağ alt VE Orta alt birlikte
        if sag_alt > 1000 and orta_alt > 1200:
            print("VIRAJ DEBUG: GÜÇLÜ SAĞ VİRAJ tespit edildi")
            return "SAGA DON"

        # SOL VİRAJ: Sol alt VE Orta alt birlikte
        if sol_alt > 1000 and orta_alt > 1200:
            print("VIRAJ DEBUG: GÜÇLÜ SOL VİRAJ tespit edildi")
            return "SOLA DON"

        # *** TEK TARAF VİRAJLARI (Üst bölgede çizgi olmamalı) ***
        ust_toplam = sol_ust + orta_ust + sag_ust

        # Sadece sağ altta çizgi var, üstte yok
        if sag_alt > 1000 and sol_alt < 500 and orta_alt < 800 and ust_toplam < 1500:
            print("VIRAJ DEBUG: SAĞ KÖŞE VİRAJ")
            return "SAGA DON"

        # Sadece sol altta çizgi var, üstte yok
        if sol_alt > 1000 and sag_alt < 500 and orta_alt < 800 and ust_toplam < 1500:
            print("VIRAJ DEBUG: SOL KÖŞE VİRAJ")
            return "SOLA DON"

        # *** ORTA ALT DOMİNANT VİRAJLAR ***
        if orta_alt > 2000:  # Çok yüksek orta alt
            if sag_alt > sol_alt * 1.5 and sag_alt > 800:
                print("VIRAJ DEBUG: SAĞ TARAF DOMINANT VİRAJ")
                return "SAGA DON"
            elif sol_alt > sag_alt * 1.5 and sol_alt > 800:
                print("VIRAJ DEBUG: SOL TARAF DOMINANT VİRAJ")
                return "SOLA DON"

        print("VIRAJ DEBUG: Viraj tespit edilmedi")
        return False

    def duz_cizgi_durumu(self, regions, angle=None):
        """DÜZ ÇİZGİ DURUMLARI - BİRİNCİ KODDAKI MÜKEMMEL YÖNTEM"""
        if self.viraj_modu_aktif:
            return None  # Viraj modundayken düz çizgi kontrolü yapma

        sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions

        # *** DÜZ GİT DURUMLARI - AÇI KONTROLÜ İLE ***
        if orta_ust > 1000 and orta_alt > 1000:
            if self.is_line_angled(angle):
                if angle < -self.angle_correction_threshold:
                    return "HAFIF SAGA DON"
                elif angle > self.angle_correction_threshold:
                    return "HAFIF SOLA DON"
            return "DUZ GIT"

        elif orta_ust > 1000 and sol_ust <= 1000 and sag_ust <= 1000:
            if self.is_line_angled(angle):
                if angle < -self.angle_correction_threshold:
                    return "HAFIF SAGA DON"
                elif angle > self.angle_correction_threshold:
                    return "HAFIF SOLA DON"
            return "DUZ GIT"

        # *** YENGEC DURUMLARI ***
        elif sol_ust > 1000 and sol_alt > 1000:
            return "SOL YENGEC"
        elif sol_ust > 1000 and orta_ust <= 1000 and sag_ust <= 1000:
            return "SOL YENGEC"
        elif sag_ust > 1000 and sag_alt > 1000:
            return "SAG YENGEC"
        elif sag_ust > 1000 and orta_ust <= 1000 and sol_ust <= 1000:
            return "SAG YENGEC"

        # *** ALT BÖLGE KONTROLÜ ***
        else:
            if orta_alt > 1000 and sol_alt <= 1000 and sag_alt <= 1000:
                return "DUZ GIT"

        return None

    def viraj_modu_yonetimi(self, regions):
        """VİRAJ MODU KARARLILIĞI YÖNETİMİ - BİRİNCİ KODDAKI MÜKEMMEL YÖNTEM"""
        viraj_tespit = self.viraj_tespit_et(regions)

        if viraj_tespit:
            # Viraj tespit edildi
            if not self.viraj_modu_aktif:
                # Viraj moduna geç
                self.viraj_modu_aktif = True
                self.viraj_modu_suresi = 0
                print(">>> VİRAJ MODU BAŞLADI <<<")
            else:
                # Viraj modu devam ediyor
                self.viraj_modu_suresi += 1

            return viraj_tespit, "VIRAJ MODU"

        else:
            # Viraj tespit edilmedi
            if self.viraj_modu_aktif:
                # Minimum süre geçti mi kontrol et
                if self.viraj_modu_suresi >= self.min_viraj_suresi:
                    # Çıkış koşullarını kontrol et
                    self.viraj_cikis_esigi -= 1
                    if self.viraj_cikis_esigi <= 0:
                        # Viraj modundan çık
                        self.viraj_modu_aktif = False
                        self.viraj_cikis_esigi = 3  # Reset
                        print(">>> VİRAJ MODU BİTTİ <<<")
                        return None, "NORMAL MODA GEÇİŞ"
                    else:
                        # Hala viraj modunda kal, son komutu tekrarla
                        return "SAGA DON" if self.son_cizgi_yonu == "SAG" else "SOLA DON", "VIRAJ MODU (DEVAM)"
                else:
                    # Minimum süre geçmedi, viraj modunda kal
                    self.viraj_modu_suresi += 1
                    return "SAGA DON" if self.son_cizgi_yonu == "SAG" else "SOLA DON", "VIRAJ MODU (MIN SÜRE)"

        return None, None

    def ana_karar_verici(self, regions, angle, center):
        """ANA KARAR VERİCİ - VİRAJ ÖNCELİKLİ SİSTEM - BİRİNCİ KODDAKI MÜKEMMEL YÖNTEM"""

        # Çizgi var mı kontrolü
        cizgi_mevcut = self.cizgi_var_mi(regions)

        if not cizgi_mevcut:
            self.cizgi_kayip_sayaci += 1
            if self.cizgi_kayip_sayaci >= self.kayip_esigi:
                return self.arama_modu_karar(), "ARAMA MODU"
            else:
                return "BEKLE", "BEKLE MODU"

        # Çizgi varsa normal işlem
        self.cizgi_kayip_sayaci = 0
        self.son_cizgi_yonunu_guncelle(regions)

        # *** 1. ÖNCE VİRAJ MODU YÖNETİMİ (EN ÖNCELİKLİ) ***
        viraj_karar, viraj_mod = self.viraj_modu_yonetimi(regions)
        if viraj_karar:
            return viraj_karar, viraj_mod

        # *** 2. SONRA DÜZ ÇİZGİ KONTROLÜ (VİRAJ MODUNDA DEĞİLSE) ***
        if not self.viraj_modu_aktif:
            duz_cizgi_sonuc = self.duz_cizgi_durumu(regions, angle)
            if duz_cizgi_sonuc:
                if "HAFIF" in duz_cizgi_sonuc:
                    return duz_cizgi_sonuc, "AÇI DÜZELTMESİ"
                elif "YENGEC" in duz_cizgi_sonuc:
                    return duz_cizgi_sonuc, "YENGEC MODU"
                else:
                    return duz_cizgi_sonuc, "NORMAL MODU"

        # *** 3. VARSAYILAN ***
        return "DUZ GIT", "NORMAL MODU"

    def calculate_ratios(self, regions):
        """Debug için oran hesaplama - BİRİNCİ KODDAN"""
        sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions

        def safe_ratio(a, b):
            return a / b if b > 0 else 0

        return {
            'sol_alt_vs_orta_alt': safe_ratio(sol_alt, orta_alt),
            'sag_alt_vs_orta_alt': safe_ratio(sag_alt, orta_alt),
            'orta_alt_vs_orta_ust': safe_ratio(orta_alt, orta_ust),
        }

    def draw_regions(self, frame):
        """Bölgeleri ve çizgileri görsel olarak çiz - BİRİNCİ KODDAKI MÜKEMMEL YÖNTEM"""
        rw_left = int(self.region_widths[0] * self.scale_factor)
        rw_mid = int((self.region_widths[0] + self.region_widths[1]) * self.scale_factor)
        rh = int(self.region_height * self.scale_factor)

        # Bölge çizgileri
        cv2.line(frame, (rw_left, 0), (rw_left, self.height), (0, 255, 0), 2)
        cv2.line(frame, (rw_mid, 0), (rw_mid, self.height), (0, 255, 0), 2)
        cv2.line(frame, (0, rh), (self.width, rh), (0, 255, 0), 2)

        # Merkez çizgisi
        cv2.line(frame, (self.width // 2, 0), (self.width // 2, self.height), (255, 0, 0), 1)

        # Bölge etiketleri
        cv2.putText(frame, "SOL UST", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "ORTA UST", (rw_left + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "SAG UST", (rw_mid + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, "SOL ALT", (10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "ORTA ALT", (rw_left + 10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    1)
        cv2.putText(frame, "SAG ALT", (rw_mid + 10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    1)

        return frame

    def draw_detected_line(self, frame, lines, center, angle):
        """Tespit edilen çizgiyi ve bilgilerini çiz - BİRİNCİ KODDAN"""
        if lines:
            for line in lines:
                x1, y1, x2, y2 = [int(x * self.scale_factor) for x in line]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if center:
                cx, cy = [int(c * self.scale_factor) for c in center]
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)

        return frame

    # YOLO ANOMALI TESPITI - IKINCI KODDAKI CALISIR HALI KORUNACAK
    def detect_anomalies(self, frame):
        """YOLO ile anomali tespiti yap"""
        anomalies = []

        if self.yolo_model is None:
            return anomalies, frame

        # Frame interval kontrolü
        self.yolo_frame_counter += 1
        if self.yolo_frame_counter % self.yolo_frame_interval != 0:
            return [], frame

        try:
            print(f"🔍 YOLO Frame {self.yolo_frame_counter} - Anomali tespiti çalışıyor...")

            # YOLO modelini çalıştır
            results = self.yolo_model(frame, verbose=False, conf=0.4, imgsz=320)

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

                            # Cooldown kontrolü
                            anomaly_key = f"{class_name}{int(x1 // 50)}{int(y1 // 50)}"

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

                            # Frame üzerine çiz
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

    # MOTOR KONTROL FONKSIYONLARI - IKINCI KODDAKI CALISIR HALI
    def execute_movement(self, hareket):
        """Hareketi motora çevir ve uygula"""
        if self.master is None:
            return

        if hareket == "DUZ GIT":
            print("🔄 duz git")
            pwm_gonder(self.master, 1, 1615)
            pwm_gonder(self.master, 2, 1600)
            pwm_gonder(self.master, 3, 1615)
            pwm_gonder(self.master, 4, 1600)
        elif hareket == "SAGA DON":
            print("🔄 saga don")
            pwm_gonder(self.master, 1, 1490)
            pwm_gonder(self.master, 2, 1650)
            pwm_gonder(self.master, 3, 1490)
            pwm_gonder(self.master, 4, 1650)
        elif hareket == "SOLA DON":
            print("🔄 sola don")
            pwm_gonder(self.master, 1, 1700)
            pwm_gonder(self.master, 2, 1490)
            pwm_gonder(self.master, 3, 1700)
            pwm_gonder(self.master, 4, 1490)
        elif hareket == "HAFIF SAGA DON":
            print("🔄 hafif saga don")
            pwm_gonder(self.master, 1, 1490)
            pwm_gonder(self.master, 2, 1600)
            pwm_gonder(self.master, 3, 1490)
            pwm_gonder(self.master, 4, 1600)
        elif hareket == "HAFIF SOLA DON":
            print("🔄 hafif sola don")
            pwm_gonder(self.master, 1, 1650)
            pwm_gonder(self.master, 2, 1490)
            pwm_gonder(self.master, 3, 1650)
            pwm_gonder(self.master, 4, 1490)
        elif hareket == "SAG YENGEC":
            print("🔄 sag yengec")
            pwm_gonder(self.master, 1, 1300)
            pwm_gonder(self.master, 2, 1700)
            pwm_gonder(self.master, 3, 1750)
            pwm_gonder(self.master, 4, 1280)
        elif hareket == "SOL YENGEC":
            print("🔄 sol yengec")
            pwm_gonder(self.master, 1, 1730)
            pwm_gonder(self.master, 2, 1300)
            pwm_gonder(self.master, 3, 1270)
            pwm_gonder(self.master, 4, 1700)
        elif hareket == "SOL ARAMA":
            print("🔄 sol arama")
            pwm_gonder(self.master, 1, 1750)
            pwm_gonder(self.master, 2, 1530)
            pwm_gonder(self.master, 3, 1750)
            pwm_gonder(self.master, 4, 1530)
        elif hareket == "SAG ARAMA":
            print("🔄 sag arama")
            pwm_gonder(self.master, 1, 1570)
            pwm_gonder(self.master, 2, 1650)
            pwm_gonder(self.master, 3, 1570)
            pwm_gonder(self.master, 4, 1650)
        elif hareket == "ORTA ARAMA":
            print("🔄 orta arama")
            pwm_gonder(self.master, 1, 1400)
            pwm_gonder(self.master, 2, 1400)
            pwm_gonder(self.master, 3, 1400)
            pwm_gonder(self.master, 4, 1400)

    def run(self):
        """Ana döngü - BİRİNCİ KODDAKI MÜKEMMEL ALGORİTMA + YOLO + IMU"""
        print("*** ÇİZGİ TAKİBİ - VİRAJ SORUN ÇÖZÜLMÜŞ VERSİYON + YOLO + IMU ***")
        print("BİRİNCİ KODDAKI SUPERIOR ALGORİTMA AKTİF!")
        print("YOLO ANOMALİ TESPİTİ AKTİF!" if self.yolo_model else "YOLO YOK - SADECE ÇİZGİ TAKİBİ")
        print("IMU DESTEKLİ DERİNLİK VE PITCH KONTROLÜ AKTİF!" if IMU_AVAILABLE and imu else "IMU YOK")
        print("Çıkmak için 'q' tuşuna basın")
        print("Duraklatmak için 'SPACE' tuşuna basın")
        print("Sıfırlamak için 'r' tuşuna basın")

        # *** İLK BAŞTA 1 SANİYE İLERİ GİT ***
        print("\n🚀 İLK BAŞTA 1 SANİYE İLERİ GİDİLİYOR - DALGA ETKİSİNİ YENEBİLMEK İÇİN...")
        if self.master is not None:
            ileri_git_baslangic = time.time()
            while time.time() - ileri_git_baslangic < 1.4:  # 1.4 saniye
                # Düz ileri git komutu
                pwm_gonder(self.master, 1, 1615)
                pwm_gonder(self.master, 2, 1600)
                pwm_gonder(self.master, 3, 1615)
                pwm_gonder(self.master, 4, 1600)
                print(f"⏰ İleri hareket: {1.4 - (time.time() - ileri_git_baslangic):.1f}s kaldı")
                time.sleep(0.1)
            print("✅ İlk hareket tamamlandı! Şimdi çizgi takibine geçiliyor...\n")
        else:
            print("⚠ Pixhawk bağlantısı yok, ilk hareket atlanıyor...\n")

        paused = False

        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("Video bitti veya okuma hatası!")
                    break

                # FPS hesaplama
                curr_time = time.time()
                self.fps = 1 / (curr_time - self.prev_time)
                self.prev_time = curr_time

                # YOLO anomali tespiti
                anomalies = []
                if self.yolo_model is not None:
                    anomalies, frame = self.detect_anomalies(frame)

                # Ön işleme - BİRİNCİ KODDAKI MÜKEMMEL YÖNTEM
                gray, thresh, small_frame = self.preprocess_frame(frame)

                # Çizgi tespiti - BİRİNCİ KODDAKI MÜKEMMEL YÖNTEM
                angle, center, lines = self.detect_line_angle(thresh)
                regions, _ = self.detect_line_position(thresh)

                # ANA KARAR VERME - BİRİNCİ KODDAKI MÜKEMMEL ALGORİTMA
                hareket, mod = self.ana_karar_verici(regions, angle, center)

                # Hareketi uygula
                self.execute_movement(hareket)

                # Debug bilgileri - BİRİNCİ KODDAN
                ratios = self.calculate_ratios(regions)

                # Görselleştirme - BİRİNCİ KODDAKI MÜKEMMEL YÖNTEM
                frame = self.draw_regions(frame)
                frame = self.draw_detected_line(frame, lines, center, angle)

                # Bilgileri formatla
                sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions
                bolge_bilgisi = f"U:[{sol_ust},{orta_ust},{sag_ust}] A:[{sol_alt},{orta_alt},{sag_alt}]"
                aci_bilgisi = f"Açı: {angle:.1f}°" if angle is not None else "Açı: Tespit edilemedi"

                # VİRAJ MODU BİLGİLERİ - BİRİNCİ KODDAN
                viraj_durum = f"Viraj Aktif: {self.viraj_modu_aktif} | Süre: {self.viraj_modu_suresi}"

                # Anomali bilgileri
                anomali_bilgisi = f"YOLO: {len(anomalies)} aktif | {self.anomaly_count} toplam" if self.yolo_model else "YOLO: YOK"

                # Ekrana yazdır
                cv2.putText(frame, f"FPS: {self.fps:.1f}", (self.width - 100, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(frame, f"Mod: {mod}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Hareket: {hareket}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, aci_bilgisi, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                cv2.putText(frame, bolge_bilgisi, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, viraj_durum, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, anomali_bilgisi, (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

                cv2.putText(frame, f"Son Yön: {self.son_cizgi_yonu} | Kayıp: {self.cizgi_kayip_sayaci}",
                            (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                if paused:
                    cv2.putText(frame, "DURAKLATILDI - SPACE ile devam et", (10, 270),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Mod çerçeveleri - BİRİNCİ KODDAN
                if self.viraj_modu_aktif:
                    cv2.rectangle(frame, (5, 5), (self.width - 5, self.height - 5), (0, 0, 255), 8)  # Kalın kırmızı
                elif mod == "ARAMA MODU":
                    cv2.rectangle(frame, (5, 5), (self.width - 5, self.height - 5), (255, 0, 0), 5)
                elif "AÇI DÜZELTMESİ" in mod:
                    cv2.rectangle(frame, (5, 5), (self.width - 5, self.height - 5), (0, 255, 255), 3)
                elif anomalies:
                    cv2.rectangle(frame, (5, 5), (self.width - 5, self.height - 5), (255, 0, 255), 4)

                # Görüntüleri göster
                cv2.imshow('Cizgi Takibi + YOLO + IMU - Superior Algoritma', frame)
                cv2.imshow('Threshold', cv2.resize(thresh, (320, 240)))

            # Tuş kontrolü
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print("Duraklatıldı" if paused else "Devam ediyor")
            elif key == ord('r'):
                self.cizgi_kayip_sayaci = 0
                self.son_cizgi_yonu = "ORTA"
                self.current_state = "NORMAL"
                self.state_counter = 0
                self.viraj_modu_aktif = False
                self.viraj_modu_suresi = 0
                self.viraj_cikis_esigi = 3
                print("Sistem sıfırlandı")

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
                print(f"Log tamamlandı: {self.anomaly_count} tespit, FPS: {self.fps:.1f}")
                print(f"Log: {self.anomaly_log_file}")
            except:
                pass

    def cleanup(self):
        """Temizlik fonksiyonu"""
        if self.cap is not None:
            self.cap.release()


def main():
    """Ana fonksiyon"""
    global shutdown_flag, depth_control_active, imu

    algorithm = None
    master = None

    try:
        print("🔧 Jetson Nano sistemi başlatılıyor...")

        # IMU başlatma
        if IMU_AVAILABLE:
            print("📡 IMU başlatılıyor...")
            try:
                imu = BNO055.BNO055(serial_port=SERIAL_IMU)
                if imu.begin():
                    time.sleep(2)
                    print("✅ IMU hazır")

                    # Referans pitch kontrolü
                    ref_pitch = referans_oku()
                    if ref_pitch is None:
                        print("⚠ Referans pitch bulunamadı, otomatik referans alınıyor...")
                        try:
                            referans_kaydet()
                        except Exception as e:
                            print(f"❌ Referans pitch kayıt hatası: {e}")
                    else:
                        print(f"📐 Mevcut referans pitch: {ref_pitch:.1f}°")
                else:
                    print("❌ IMU başlatılamadı")
                    imu = None
            except Exception as e:
                print(f"❌ IMU hatası: {e}")
                imu = None
        else:
            print("⚠ IMU kütüphanesi yok")
            imu = None

        # Pixhawk bağlantısı
        print("🚁 Pixhawk bağlantısı deneniyor...")
        master = baglanti_kur()

        if master:
            arm_motors(master)
            time.sleep(1)

            # Derinlik ve pitch kontrol başlat
            depth_thread = threading.Thread(target=depth_control_thread, args=(master,))
            depth_thread.daemon = True
            depth_thread.start()
            print("🌊 IMU destekli derinlik ve pitch kontrol başlatıldı")
        else:
            print("⚠ Pixhawk bağlanamadı, sadece kamera modu")

        # Model yolu belirleme
        possible_paths = [
            "best.pt",
            "./best.pt",
            "/home/nova/Downloads/best.pt",
            "models/best.pt"
        ]

        model_path = "best.pt"  # Varsayılan
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f"✅ Model bulundu: {path}")
                break

        # Algoritma başlat - BİRİNCİ KODDAKI SUPERIOR ALGORİTMA İLE
        print("🚀 BİRİNCİ KODDAKI SUPERIOR ALGORİTMA + YOLO + IMU AKTİF!")
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
                tum_motorlari_durdur(master)
                print("🔴 Motorlar durduruldu")
            except Exception as e:
                print(f"Motor durdurma hatası: {e}")

        if algorithm:
            try:
                algorithm.cleanup()
            except Exception as e:
                print(f"Temizlik hatası: {e}")

        print("✅ Program güvenli şekilde sonlandı")


if __name__ == "__main__":
    main()