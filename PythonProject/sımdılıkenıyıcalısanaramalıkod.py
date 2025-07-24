import cv2
import numpy as np
import math
import time


class LineFollowingAlgorithm:
    def __init__(self, video_path=None):
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
        else:
            # Kamera ayarlarını optimize et
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Video boyutlarını al
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # İşleme için küçültülmüş boyutlar
        self.process_width = 320
        self.scale_factor = self.width / self.process_width

        # *** DEĞİŞTİRİLEN KISIM: Ayarlanabilir bölge genişlikleri ***
        self.region_ratios = [0.35, 0.30, 0.35]  # Sol, Orta, Sağ oranları (toplam 1.0 olmalı) # sadece buralarla oynayacaksun kübra sutun genıslıklerı ıcın
        self.region_widths = [
            int(self.process_width * self.region_ratios[0]),  # Sol genişlik
            int(self.process_width * self.region_ratios[1]),  # Orta genişlik
            int(self.process_width * self.region_ratios[2])   # Sağ genişlik
        ]
        self.region_height = int(self.height * 0.6 / self.scale_factor)

        # Çizgi rengi eşiği (siyah çizgi için)
        self.lower_threshold = 0
        self.upper_threshold = 50

        # *** DÜZELTİLMİŞ PARAMETRELER ***
        self.max_allowed_angle = 15
        self.min_line_length = 25
        self.angle_correction_threshold = 20

        # *** VİRAJ TESPİTİ PARAMETRELERİ ***
        self.orta_ust_viraj_killer = 500
        self.viraj_pixel_threshold = 1000
        self.viraj_dominance_ratio = 1.05

        # *** NORMAL İŞLEM PARAMETRELERİ ***
        self.normal_pixel_esigi = 800
        self.minimum_pixel_esigi = 1000
        self.dominance_ratio = 1.3

        # FPS hesaplama
        self.prev_time = time.time()
        self.fps = 0

        # ARAMA MODU PARAMETRELERİ
        self.son_cizgi_yonu = "ORTA"
        self.cizgi_kayip_sayaci = 0
        self.kayip_esigi = 5

        # DURUM KONTROLÜ
        self.current_state = "NORMAL"
        self.state_counter = 0

    def cizgi_var_mi(self, regions):
        """Çizgi var mı yok mu kontrol et"""
        return any(region > self.minimum_pixel_esigi for region in regions)

    def son_cizgi_yonunu_guncelle(self, regions):
        """Son görülen çizginin yönünü güncelle"""
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
        """Arama modunda hangi yöne gidileceğini belirle"""
        if self.son_cizgi_yonu == "SOL":
            return "SOL ARAMA"
        elif self.son_cizgi_yonu == "SAG":
            return "SAG ARAMA"
        else:
            return "ORTA ARAMA"

    def preprocess_frame(self, frame):
        """Görüntüyü küçült ve ön işleme yap"""
        small_frame = cv2.resize(frame, (self.process_width, int(self.height / self.scale_factor)))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.upper_threshold, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        return gray, thresh, small_frame

    def detect_line_angle(self, gray_frame):
        """Optimize edilmiş çizgi açısı tespiti"""
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
        """Optimize edilmiş bölge tespiti - Yeni bölge genişlikleriyle"""
        upper_half = thresh[0:self.region_height, :]
        lower_half = thresh[self.region_height:, :]

        # Yeni bölge sınırlarına göre hesaplama
        upper_regions = [
            np.count_nonzero(upper_half[:, 0:self.region_widths[0]]),  # Sol üst
            np.count_nonzero(upper_half[:, self.region_widths[0]:self.region_widths[0]+self.region_widths[1]]),  # Orta üst
            np.count_nonzero(upper_half[:, self.region_widths[0]+self.region_widths[1]:])  # Sağ üst
        ]

        lower_regions = [
            np.count_nonzero(lower_half[:, 0:self.region_widths[0]]),  # Sol alt
            np.count_nonzero(lower_half[:, self.region_widths[0]:self.region_widths[0]+self.region_widths[1]]),  # Orta alt
            np.count_nonzero(lower_half[:, self.region_widths[0]+self.region_widths[1]:])  # Sağ alt
        ]

        return upper_regions + lower_regions, thresh

    def is_line_angled(self, angle):
        """Çizginin açılı olup olmadığını kontrol eder"""
        if angle is None:
            return False
        return abs(angle) > self.angle_correction_threshold

    def duz_cizgi_durumu(self, regions, angle=None):
        """1. FONKSİYON: DÜZ ÇİZGİ DURUMLARI"""
        sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions

        # *** DÜZ GİT DURUMLARI - AÇI KONTROLÜ İLE ***
        # Orta üst VE orta alt varsa → DÜZ GİT (açı kontrolü ile)
        if orta_ust > 1000 and orta_alt > 1000:
            if self.is_line_angled(angle):
                if angle < -self.angle_correction_threshold:
                    return "HAFIF SAGA DON(açı kontrolü ile)"
                elif angle > self.angle_correction_threshold:
                    return "HAFIF SOLA DON(açı kontrolü ile)"
            return "DUZ GIT"

        # Sadece orta üst varsa → DÜZ GİT (açı kontrolü ile)
        elif orta_ust > 1000 and sol_ust <= 1000 and sag_ust <= 1000:
            if self.is_line_angled(angle):
                if angle < -self.angle_correction_threshold:
                    return "HAFİF SAGA DON(açı kontrolü ile)"
                elif angle > self.angle_correction_threshold:
                    return "HAFİF SOLA DON(açı kontrolü ile)"
            return "DUZ GIT"

        # *** SOL YENGEC DURUMLARI ***
        # Sol üst VE sol alt varsa → SOL YENGEC
        elif sol_ust > 1000 and sol_alt > 1000:
            return "SOL YENGEC"

        # Sadece sol üst varsa → SOL YENGEC
        elif sol_ust > 1000 and orta_ust <= 1000 and sag_ust <= 1000:
            return "SOL YENGEC"

        # *** SAĞ YENGEC DURUMLARI ***
        # Sağ üst VE sağ alt varsa → SAĞ YENGEC
        elif sag_ust > 1000 and sag_alt > 1000:
            return "SAG YENGEC"

        # Sadece sağ üst varsa → SAĞ YENGEC
        elif sag_ust > 1000 and orta_ust <= 1000 and sol_ust <= 1000:
            return "SAG YENGEC"

        # *** ALT BÖLGE KONTROLÜ (Fallback) ***
        # Üst bölgede hiçbir şey yoksa alt bölgeye bak
        else:
            lower_regions = [sol_alt, orta_alt, sag_alt]
            max_index = np.argmax(lower_regions)
            max_value = lower_regions[max_index]

            if max_value > 1000:
                if max_index == 0:
                    return "SOL YENGEC"
                elif max_index == 1:
                    return "DUZ GIT"
                else:
                    return "SAG YENGEC"

        return None  # Hiçbir durum uymazsa None döner

    def viraj_durumu(self, regions):
        """2. FONKSİYON: VİRAJ DURUMLARI"""
        sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions

        print(f"DEBUG VIRAJ: orta_ust={orta_ust}, orta_alt={orta_alt}, sag_alt={sag_alt}, sol_alt={sol_alt}")

        # *** ÇAKIŞMA ÖNLEYİCİ - Orta üst çok yüksekse viraj değil ***
        if orta_ust > self.orta_ust_viraj_killer:
            print(f"DEBUG: Orta üst çok yüksek - Viraj değil")
            return None

        # *** SAĞ VİRAJ: Sağ alt VE Orta alt varsa ***
        if sag_alt > 1000 and orta_alt > 1500:
            print("DEBUG: SAĞ VİRAJ (Sağ alt + Orta alt)")
            return "SAGA DON"

        # *** SOL VİRAJ: Sol alt VE Orta alt varsa ***
        elif sol_alt > 1000 and orta_alt > 1500:
            print("DEBUG: SOL VİRAJ (Sol alt + Orta alt)")
            return "SOLA DON"

        # *** EK VİRAJ KONTROLLERI ***
        # Sadece orta alt varsa ve bir taraf belirgin
        elif orta_alt > 1500:
            if sag_alt < 500 and sol_alt > 800:  # Sağ taraf boş, sol tarafta çizgi
                print("DEBUG: SOL VİRAJ (Sol taraf dominant)")
                return "SOLA DON"
            elif sol_alt < 500 and sag_alt > 800:  # Sol taraf boş, sağ tarafta çizgi
                print("DEBUG: SAĞ VİRAJ (Sağ taraf dominant)")
                return "SAGA DON"

        print("DEBUG: Viraj koşulları sağlanmadı")
        return None

    def calculate_ratios(self, regions):
        """Debug için oran hesaplama"""
        sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions

        def safe_ratio(a, b):
            return a / b if b > 0 else 0

        return {
            'sol_alt_vs_orta_alt': safe_ratio(sol_alt, orta_alt),
            'sag_alt_vs_orta_alt': safe_ratio(sag_alt, orta_alt),
            'orta_alt_vs_orta_ust': safe_ratio(orta_alt, orta_ust),
        }

    def ana_karar_verici(self, regions, angle, center):
        """ANA KARAR VERİCİ"""

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

        # *** 2 FONKSİYON SİSTEMİ ***

        # 1. ÖNCE VİRAJ KONTROLÜ
        viraj_sonuc = self.viraj_durumu(regions)
        if viraj_sonuc:
            return viraj_sonuc, "VIRAJ MODU"

        # 2. SONRA DÜZ ÇİZGİ KONTROLÜ
        duz_cizgi_sonuc = self.duz_cizgi_durumu(regions, angle)
        if duz_cizgi_sonuc:
            if "HAFIF" in duz_cizgi_sonuc:
                return duz_cizgi_sonuc, "AÇI DÜZELTMESİ"
            elif "YENGEC" in duz_cizgi_sonuc:
                return duz_cizgi_sonuc, "YENGEC MODU"
            else:
                return duz_cizgi_sonuc, "NORMAL MODU"

        # 3. VARSAYILAN
        return "DUZ GIT", "NORMAL MODU"

    def draw_regions(self, frame):
        """Bölgeleri ve çizgileri görsel olarak çiz - Yeni bölge genişlikleriyle"""
        # Orijinal boyutta bölge sınırlarını hesapla
        rw_left = int(self.region_widths[0] * self.scale_factor)
        rw_mid = int((self.region_widths[0] + self.region_widths[1]) * self.scale_factor)
        rh = int(self.region_height * self.scale_factor)

        # Bölge çizgileri
        cv2.line(frame, (rw_left, 0), (rw_left, self.height), (0, 255, 0), 2)  # Sol-Orta ayrımı
        cv2.line(frame, (rw_mid, 0), (rw_mid, self.height), (0, 255, 0), 2)    # Orta-Sağ ayrımı
        cv2.line(frame, (0, rh), (self.width, rh), (0, 255, 0), 2)             # Üst-Alt ayrımı

        # Merkez çizgisi
        cv2.line(frame, (self.width // 2, 0), (self.width // 2, self.height), (255, 0, 0), 1)

        # Bölge etiketleri (konumlar yeni genişliklere göre ayarlandı)
        cv2.putText(frame, "SOL UST", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "ORTA UST", (rw_left + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "SAG UST", (rw_mid + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, "SOL ALT", (10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "ORTA ALT", (rw_left + 10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "SAG ALT", (rw_mid + 10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def draw_detected_line(self, frame, lines, center, angle):
        """Tespit edilen çizgiyi ve bilgilerini çiz"""
        if lines:
            for line in lines:
                x1, y1, x2, y2 = [int(x * self.scale_factor) for x in line]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if center:
                cx, cy = [int(c * self.scale_factor) for c in center]
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)

        return frame

    def run(self):
        """Ana döngü"""
        print("*** ÇİZGİ TAKİBİ ***")
        print("Çıkmak için 'q' tuşuna basın")
        print("Duraklatmak için 'SPACE' tuşuna basın")
        print("Sıfırlamak için 'r' tuşuna basın")

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

                # Ön işleme
                gray, thresh, small_frame = self.preprocess_frame(frame)

                # Çizgi tespiti
                angle, center, lines = self.detect_line_angle(thresh)
                regions, _ = self.detect_line_position(thresh)

                # ANA KARAR VERME
                hareket, mod = self.ana_karar_verici(regions, angle, center)

                # Debug bilgileri
                ratios = self.calculate_ratios(regions)

                # Görselleştirme
                frame = self.draw_regions(frame)
                frame = self.draw_detected_line(frame, lines, center, angle)

                # Bilgileri formatla
                sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions
                bolge_bilgisi = f"U:[{sol_ust},{orta_ust},{sag_ust}] A:[{sol_alt},{orta_alt},{sag_alt}]"
                aci_bilgisi = f"Açı: {angle:.1f}°" if angle is not None else "Açı: Tespit edilemedi"

                # Ekrana yazdır
                cv2.putText(frame, f"FPS: {self.fps:.1f}", (self.width - 100, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(frame, f"Mod: {mod}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Hareket: {hareket}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, aci_bilgisi, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                cv2.putText(frame, bolge_bilgisi, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Debug ratios
                cv2.putText(frame,
                            f"Ratios: S/O:{ratios['sol_alt_vs_orta_alt']:.1f} Sg/O:{ratios['sag_alt_vs_orta_alt']:.1f}",
                            (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"Son Yön: {self.son_cizgi_yonu} | Kayıp: {self.cizgi_kayip_sayaci}",
                            (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                # Viraj tespiti bilgisi
                viraj_sonuc = self.viraj_durumu(regions)
                viraj_text = f"VIRAJ: {viraj_sonuc}" if viraj_sonuc else "VIRAJ YOK"
                viraj_color = (0, 0, 255) if viraj_sonuc else (0, 255, 0)
                cv2.putText(frame, viraj_text, (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, viraj_color, 2)

                if paused:
                    cv2.putText(frame, "DURAKLATILDI - SPACE ile devam et", (10, 270),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Mod çerçeveleri
                if mod == "VIRAJ MODU":
                    cv2.rectangle(frame, (5, 5), (self.width - 5, self.height - 5), (0, 0, 255), 5)
                elif mod == "ARAMA MODU":
                    cv2.rectangle(frame, (5, 5), (self.width - 5, self.height - 5), (255, 0, 0), 5)

                # Görüntüleri göster
                cv2.imshow('Cizgi Takibi', frame)
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
                print("Sistem sıfırlandı")

        # Temizlik
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        video_path ="C:/Users/user/Downloads/video3.mp4" #0   # = webcam, video dosyası için path verin

        algorithm = LineFollowingAlgorithm(video_path)
        algorithm.run()
    except KeyboardInterrupt:
        print("\nProgram durduruldu.")
    except Exception as e:
        print(f"Hata: {e}")
        print("Video dosyası yolunu kontrol edin.")