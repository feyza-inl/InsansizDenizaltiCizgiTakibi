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
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Düşük çözünürlük
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)  # FPS ayarı
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer boyutunu azalt

        # Video boyutlarını al
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # İşleme için küçültülmüş boyutlar
        self.process_width = 320  # İşleme boyutu
        self.scale_factor = self.width / self.process_width

        # Bölge ayarları (küçültülmüş boyuta göre)
        self.region_width = self.process_width // 3
        self.region_height = int(self.height * 0.6 / self.scale_factor)

        # Çizgi rengi eşiği (siyah çizgi için)
        self.lower_threshold = 0
        self.upper_threshold = 50

        # AÇI KONTROL PARAMETRELERİ
        self.max_allowed_angle = 20  # Kabul edilebilir maksimum açı (derece)
        self.min_line_length = 30  # Daha kısa çizgiler için (optimize)
        self.angle_correction_threshold = 10  # Bu açıdan fazlası düzeltme gerektirir

        # FPS hesaplama
        self.prev_time = time.time()
        self.fps = 0

        # YENİ EKLENEN: ARAMA MODU PARAMETRELERİ
        self.son_cizgi_yonu = "ORTA"  # Son görülen çizginin yönü
        self.cizgi_kayip_sayaci = 0  # Çizgi kaybı sayacı
        self.kayip_esigi = 3  # Kaç frame çizgi görülmezse arama moduna geçsin
        self.minimum_pixel_esigi = 500  # Çizgi var sayılması için minimum pixel sayısı

    # YENİ EKLENEN: ÇİZGİ VAR MI KONTROLÜ
    def cizgi_var_mi(self, regions):
        """Çizgi var mı yok mu kontrol et"""
        return any(region > self.minimum_pixel_esigi for region in regions)

    # YENİ EKLENEN: SON ÇİZGİ YÖNÜNÜ GÜNCELLE
    def son_cizgi_yonunu_guncelle(self, regions):
        """Son görülen çizginin yönünü güncelle"""
        sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions

        # En yüksek pixel sayısına sahip bölgeyi bul
        max_value = max(regions)
        max_index = regions.index(max_value)

        # İndekse göre yönü belirle
        if max_index in [0, 3]:  # Sol üst veya sol alt
            self.son_cizgi_yonu = "SOL"
        elif max_index in [1, 4]:  # Orta üst veya orta alt
            self.son_cizgi_yonu = "ORTA"
        elif max_index in [2, 5]:  # Sağ üst veya sağ alt
            self.son_cizgi_yonu = "SAG"

    # YENİ EKLENEN: ARAMA MODU KARAR FONKSİYONU
    def arama_modu_karar(self):
        """Arama modunda hangi yöne gidileceğini belirle"""
        if self.son_cizgi_yonu == "SOL":
            return "SOL ARAMA"  # Sola doğru salyangoz hareketi
        elif self.son_cizgi_yonu == "SAG":
            return "SAG ARAMA"  # Sağa doğru salyangoz hareketi
        else:
            return "ORTA ARAMA"  # Orta bölgede kaybolmuşsa hafif zigzag

    def preprocess_frame(self, frame):
        """Görüntüyü küçült ve ön işleme yap"""
        # Boyutu küçült
        small_frame = cv2.resize(frame, (self.process_width, int(self.height / self.scale_factor)))
        # Gri tonlama
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        # Threshold
        _, thresh = cv2.threshold(gray, self.upper_threshold, 255, cv2.THRESH_BINARY_INV)
        # Gürültü azaltma
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        return gray, thresh, small_frame

    def detect_line_angle(self, gray_frame):
        """Optimize edilmiş çizgi açısı tespiti"""
        # HoughLinesP parametreleri optimize edildi
        lines = cv2.HoughLinesP(gray_frame, 1, np.pi / 180,
                                threshold=30,  # Daha düşük eşik
                                minLineLength=self.min_line_length,
                                maxLineGap=20)  # Daha büyük boşluk

        if lines is not None:
            # En uzun 2 çizgiyi al
            lines = sorted(lines, key=lambda x: np.linalg.norm(x[0][2:] - x[0][:2]), reverse=True)[:2]

            angles = []
            centers = []
            valid_lines = []

            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                if length > self.min_line_length:
                    angle = math.degrees(math.atan2(x2 - x1, y2 - y1))
                    # Açı normalizasyonu
                    if angle > 90:
                        angle -= 180
                    elif angle < -90:
                        angle += 180

                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    angles.append(angle)
                    centers.append(center)
                    valid_lines.append(line[0])

            if angles:
                # Ortalama açı ve merkez
                avg_angle = np.mean(angles)
                avg_center = (int(np.mean([c[0] for c in centers])),
                              int(np.mean([c[1] for c in centers])))
                return avg_angle, avg_center, valid_lines

        return None, None, None

    def detect_line_position(self, thresh):
        """Optimize edilmiş bölge tespiti"""
        # Üst ve alt bölgeler
        upper_half = thresh[0:self.region_height, :]
        lower_half = thresh[self.region_height:, :]

        # Hızlı bölge analizi
        upper_regions = [
            np.count_nonzero(upper_half[:, i * self.region_width:(i + 1) * self.region_width])
            for i in range(3)
        ]

        lower_regions = [
            np.count_nonzero(lower_half[:, i * self.region_width:(i + 1) * self.region_width])
            for i in range(3)
        ]

        return upper_regions + lower_regions, thresh

    def is_line_angled(self, angle):
        """Çizginin açılı olup olmadığını kontrol eder"""
        if angle is None:
            return False
        return abs(angle) > self.angle_correction_threshold

    def viraj_tespiti(self, regions):
        """Viraj var mı yok mu kontrol et"""
        sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions

        # KILIT KURAL: Orta alt + Orta üst varsa → DÜZ GİT (viraj değil)
        if orta_alt > 1000 and orta_ust > 1000:
            return False

        # Viraj durumları
        if orta_alt > 1000:
            if sag_alt > 1000:
                return True  # Sağa dön virajı
            elif sol_alt > 1000:
                return True  # Sola dön virajı

        return False

    def viraj_fonksiyonu(self, regions):
        """Viraj tespiti ve hareket kararı"""
        sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions

        if orta_alt > 1000 and sag_alt > 1000:
            return "SAGA DON"
        elif orta_alt > 1000 and sol_alt > 1000:
            return "SOLA DON"
        else:
            return "VIRAJ TESPIT EDILEMEDI"

    def duz_cizgi_fonksiyonu(self, regions, angle=None, line_center=None):
        """Açı kontrolü eklenmiş düz çizgi fonksiyonu"""
        sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions

        # AÇI KONTROL BLOĞU
        if orta_alt > 1000 and orta_ust > 1000:
            if self.is_line_angled(angle):
                if angle < -self.angle_correction_threshold:
                    return "SAGA DON (AÇI DÜZELTMESİ)"  # Çizgi sağa eğik, sola dön
                elif angle > self.angle_correction_threshold:
                    return "SOLA DON (AÇI DÜZELTMESİ)"  # Çizgi sola eğik, sağa dön
            else:
                return "DUZ GIT"  # Çizgi düz, normal ilerle

        # ÜST BÖLGE KONTROLÜ
        if sol_ust > 1000 and orta_ust <= 1000 and sag_ust <= 1000:
            return "SOL YENGEC"
        elif sag_ust > 1000 and orta_ust <= 1000 and sol_ust <= 1000:
            return "SAG YENGEC"
        elif orta_ust > 1000 and sol_ust <= 1000 and sag_ust <= 1000:
            return "DUZ GIT"

        # MEVCUT MANTIK - Orta alt + Orta üst = DÜZ GİT
        if orta_alt > 1000 and orta_ust > 1000:
            return "DUZ GIT"

        # Sadece alt bölgelere bak
        lower_regions = [sol_alt, orta_alt, sag_alt]
        max_index = np.argmax(lower_regions)

        if max_index == 0:
            return "SOL YENGEC"
        elif max_index == 1:
            return "DUZ GIT"
        else:
            return "SAG YENGEC"

    def draw_regions(self, frame):
        """Bölgeleri ve çizgileri görsel olarak çiz"""
        # Bölge çizgileri (orijinal boyutta)
        rw = int(self.region_width * self.scale_factor)
        rh = int(self.region_height * self.scale_factor)
        cv2.line(frame, (rw, 0), (rw, self.height), (0, 255, 0), 2)
        cv2.line(frame, (rw * 2, 0), (rw * 2, self.height), (0, 255, 0), 2)
        cv2.line(frame, (0, rh), (self.width, rh), (0, 255, 0), 2)

        # Merkez çizgisi
        cv2.line(frame, (self.width // 2, 0), (self.width // 2, self.height), (255, 0, 0), 1)

        # Bölge etiketleri
        cv2.putText(frame, "SOL UST", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "ORTA UST", (rw + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "SAG UST", (rw * 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, "SOL ALT", (10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "ORTA ALT", (rw + 10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "SAG ALT", (rw * 2 + 10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
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
        """Ana döngü - ARAMA MODU EKLENDİ"""
        print("*** OPTİMİZE ÇİZGİ TAKİBİ BAŞLATILDI (ARAMA MODLU) ***")
        print("Çıkmak için 'q' tuşuna basın")
        print("Duraklatmak için 'SPACE' tuşuna basın")
        print("Arama modunu sıfırlamak için 'r' tuşuna basın")

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

                # YENİ EKLENEN: ÇİZGİ VAR MI KONTROLÜ
                cizgi_mevcut = self.cizgi_var_mi(regions)

                # YENİ EKLENEN: ARAMA MODU KONTROLÜ
                if cizgi_mevcut:
                    # Çizgi varsa normal işlem
                    self.cizgi_kayip_sayaci = 0  # Sayacı sıfırla
                    self.son_cizgi_yonunu_guncelle(regions)  # Son yönü güncelle

                    # Viraj kontrolü
                    is_viraj = self.viraj_tespiti(regions)

                    # Hareket kararı ver
                    if is_viraj:
                        hareket = self.viraj_fonksiyonu(regions)
                        mod = "VIRAJ MODU"
                    else:
                        hareket = self.duz_cizgi_fonksiyonu(regions, angle, center)
                        mod = "DUZ CIZGI MODU"
                else:
                    # Çizgi yoksa arama moduna geç
                    self.cizgi_kayip_sayaci += 1

                    if self.cizgi_kayip_sayaci >= self.kayip_esigi:
                        hareket = self.arama_modu_karar()
                        mod = "ARAMA MODU"
                    else:
                        hareket = "BEKLE"  # Birkaç frame daha bekle
                        mod = "BEKLE MODU"

                # Görselleştirme
                frame = self.draw_regions(frame)
                frame = self.draw_detected_line(frame, lines, center, angle)

                # Bilgileri formatla
                sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions
                bolge_bilgisi = f"U:[{sol_ust}, {orta_ust}, {sag_ust}] A:[{sol_alt}, {orta_alt}, {sag_alt}]"
                aci_bilgisi = f"Açı: {angle:.1f}°" if angle is not None else "Açı: Tespit edilemedi"

                # Ekrana yazdır
                cv2.putText(frame, f"FPS: {self.fps:.1f}", (self.width - 100, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(frame, f"Mod: {mod}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Hareket: {hareket}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, aci_bilgisi, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                cv2.putText(frame, bolge_bilgisi, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # YENİ EKLENEN: ARAMA MODU BİLGİLERİ
                cv2.putText(frame, f"Son Cizgi Yonu: {self.son_cizgi_yonu}", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"Kayip Sayaci: {self.cizgi_kayip_sayaci}", (10, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"Cizgi Mevcut: {'EVET' if cizgi_mevcut else 'HAYIR'}", (10, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if cizgi_mevcut else (0, 0, 255), 1)

                if paused:
                    cv2.putText(frame, "DURAKLATILDI - SPACE ile devam et", (10, 270),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Arama modundaysa kırmızı çerçeve çiz
                if mod == "ARAMA MODU":
                    cv2.rectangle(frame, (5, 5), (self.width - 5, self.height - 5), (0, 0, 255), 5)

                # Görüntüleri göster
                cv2.imshow('Optimize Cizgi Takibi (Arama Modlu)', frame)
                cv2.imshow('Threshold', cv2.resize(thresh, (320, 240)))

            # Tuş kontrolü
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print("Duraklatıldı" if paused else "Devam ediyor")
            elif key == ord('r'):  # YENİ EKLENEN: ARAMA MODU SIFIRLAMA
                self.cizgi_kayip_sayaci = 0
                self.son_cizgi_yonu = "ORTA"
                print("Arama modu sıfırlandı")

        # Temizlik
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        # Video dosyası yolunu buraya yazın
        video_path = "C:/Users/user/Downloads/video3.mp4"

        algorithm = LineFollowingAlgorithm(video_path)
        algorithm.run()
    except KeyboardInterrupt:
        print("\nProgram durduruldu.")
    except Exception as e:
        print(f"Hata: {e}")
        print("Video dosyası yolunu kontrol edin.")