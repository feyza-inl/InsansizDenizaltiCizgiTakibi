import cv2
import numpy as np
import math
import time


class LineFollowingAlgorithm:
    def _init_(self, video_path=None):
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

        # Ayarlanabilir bölge genişlikleri
        self.region_ratios = [0.35, 0.30, 0.35]  # Sol, Orta, Sağ oranları
        self.region_widths = [
            int(self.process_width * self.region_ratios[0]),  # Sol genişlik
            int(self.process_width * self.region_ratios[1]),  # Orta genişlik
            int(self.process_width * self.region_ratios[2])  # Sağ genişlik
        ]
        self.region_height = int(self.height * 0.6 / self.scale_factor)

        # Çizgi rengi eşiği (siyah çizgi için)
        self.lower_threshold = 0
        self.upper_threshold = 50

        # *** DÜZELTILMIŞ PARAMETRELER - VIRAJ ÖNCELIKLI SISTEM ***
        self.max_allowed_angle = 15
        self.min_line_length = 25
        self.angle_correction_threshold = 20  # Arttırıldı, daha az hassas

        # *** VIRAJ MODU KARARLILIĞI İÇİN YENİ PARAMETRELER ***
        self.viraj_modu_aktif = False
        self.viraj_modu_suresi = 0
        self.min_viraj_suresi = 10  # En az 8 frame viraj modunda kalsın
        self.viraj_cikis_esigi = 3  # 3 frame boyunca viraj tespit edilmezse çık

        # VİRAJ TESPİTİ PARAMETRELERİ
        self.orta_ust_viraj_killer = 500
        self.viraj_pixel_threshold = 1000
        self.viraj_dominance_ratio = 1.05

        # NORMAL İŞLEM PARAMETRELERİ
        self.normal_pixel_esigi = 800
        self.minimum_pixel_esigi = 600
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
        """Optimize edilmiş bölge tespiti"""
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
        """Çizginin açılı olup olmadığını kontrol eder - SADECE VİRAJ MODUNDA DEĞİLSE"""
        if angle is None or self.viraj_modu_aktif:
            return False
        return abs(angle) > self.angle_correction_threshold

    def viraj_tespit_et(self, regions):
        """ÖNCELIKLI VIRAJ TESPIT FONKSIYONU"""
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
        """DÜZ ÇİZGİ DURUMLARI - VİRAJ MODUNDA DEĞİLSE"""
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
        """VİRAJ MODU KARARLILIĞI YÖNETİMİ"""
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
        """ANA KARAR VERİCİ - VİRAJ ÖNCELİKLİ SİSTEM"""

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
        """Debug için oran hesaplama"""
        sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions

        def safe_ratio(a, b):
            return a / b if b > 0 else 0

        return {
            'sol_alt_vs_orta_alt': safe_ratio(sol_alt, orta_alt),
            'sag_alt_vs_orta_alt': safe_ratio(sag_alt, orta_alt),
            'orta_alt_vs_orta_ust': safe_ratio(orta_alt, orta_ust),
        }

    def draw_regions(self, frame):
        """Bölgeleri ve çizgileri görsel olarak çiz"""
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
        print("*** ÇİZGİ TAKİBİ - VİRAJ SORUN ÇÖZÜLMÜŞ VERSİYON ***")
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

                # *** VİRAJ MODU BİLGİLERİ ***
                viraj_durum = f"Viraj Aktif: {self.viraj_modu_aktif} | Süre: {self.viraj_modu_suresi}"

                # Ekrana yazdır
                cv2.putText(frame, f"FPS: {self.fps:.1f}", (self.width - 100, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(frame, f"Mod: {mod}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Hareket: {hareket}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, aci_bilgisi, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                cv2.putText(frame, bolge_bilgisi, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, viraj_durum, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                cv2.putText(frame, f"Son Yön: {self.son_cizgi_yonu} | Kayıp: {self.cizgi_kayip_sayaci}",
                            (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                if paused:
                    cv2.putText(frame, "DURAKLATILDI - SPACE ile devam et", (10, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Mod çerçeveleri
                if self.viraj_modu_aktif:
                    cv2.rectangle(frame, (5, 5), (self.width - 5, self.height - 5), (0, 0, 255), 8)  # Kalın kırmızı
                elif mod == "ARAMA MODU":
                    cv2.rectangle(frame, (5, 5), (self.width - 5, self.height - 5), (255, 0, 0), 5)
                elif "AÇI DÜZELTMESİ" in mod:
                    cv2.rectangle(frame, (5, 5), (self.width - 5, self.height - 5), (0, 255, 255), 3)

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
                self.viraj_modu_aktif = False
                self.viraj_modu_suresi = 0
                self.viraj_cikis_esigi = 3
                print("Sistem sıfırlandı")

        # Temizlik
        self.cap.release()
        cv2.destroyAllWindows()


if _name_ == "_main_":
    try:
        video_path = "C:/Users/user/Downloads/video7.mp4"  # 0 = webcam, video dosyası için path verin

        algorithm = LineFollowingAlgorithm(video_path)
        algorithm.run()
    except KeyboardInterrupt:
        print("\nProgram durduruldu.")
    except Exception as e:
        print(f"Hata: {e}")
        print("Video dosyası yolunu kontrol edin.")