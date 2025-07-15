import cv2
import numpy as np
import math


class LineFollowingAlgorithm:
    def __init__(self, video_path=None):
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
        else:
            self.cap = cv2.VideoCapture(0)

        # Video boyutlarını al
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Ekranı 6 bölgeye ayırma (3 dikey x 2 yatay)
        self.region_width = self.width // 3
        self.region_height = int(self.height * 0.6)

        # Çizgi rengi eşiği (siyah çizgi için)
        self.lower_threshold = 0
        self.upper_threshold = 50

        # *** YENİ: AÇI KONTROL PARAMETRELERİ ***
        self.max_allowed_angle = 20  # Kabul edilebilir maksimum açı (derece)
        self.min_line_length = 50  # Açı hesaplama için minimum çizgi uzunluğu
        self.angle_correction_threshold = 10  # Bu açıdan fazlası düzeltme gerektirir

    def detect_line_angle(self, frame):
        """Çizginin açısını HoughLinesP ile tespit eder"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.upper_threshold, 255, cv2.THRESH_BINARY_INV)

        # HoughLinesP ile çizgileri tespit et
        lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, threshold=50,
                                minLineLength=self.min_line_length, maxLineGap=10)

        if lines is not None:
            # En uzun çizgiyi bul
            longest_line = None
            max_length = 0

            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if length > max_length:
                    max_length = length
                    longest_line = line[0]

            if longest_line is not None:
                x1, y1, x2, y2 = longest_line

                # Çizginin açısını hesapla (dikey eksene göre)
                angle = math.degrees(math.atan2(x2 - x1, y2 - y1))

                # Açıyı -90 ile +90 arasında normalize et
                if angle > 90:
                    angle = angle - 180
                elif angle < -90:
                    angle = angle + 180

                # Çizginin orta noktasını bul
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                return angle, (center_x, center_y), longest_line

        return None, None, None

    def detect_line_position(self, frame):
        """Çizginin hangi bölgede olduğunu tespit eder - 6 bölge"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.upper_threshold, 255, cv2.THRESH_BINARY_INV)

        # ÜST YARIM
        upper_half = thresh[0:self.region_height, :]
        # ALT YARIM
        lower_half = thresh[self.region_height:, :]

        # Üst yarım için bölge analizi
        upper_regions = []
        for i in range(3):
            start_x = i * self.region_width
            end_x = (i + 1) * self.region_width
            region = upper_half[:, start_x:end_x]
            white_pixels = np.sum(region == 255)
            upper_regions.append(white_pixels)

        # Alt yarım için bölge analizi
        lower_regions = []
        for i in range(3):
            start_x = i * self.region_width
            end_x = (i + 1) * self.region_width
            region = lower_half[:, start_x:end_x]
            white_pixels = np.sum(region == 255)
            lower_regions.append(white_pixels)

        all_regions = upper_regions + lower_regions
        return all_regions, thresh

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
        """*** GELİŞTİRİLMİŞ: Açı kontrolü eklenmiş düz çizgi fonksiyonu ***"""
        sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions

        # *** YENİ: AÇI KONTROL BLOĞU ***
        # Eğer çizgi orta bölgelerde var ama açılıysa düzeltme yap
        if orta_alt > 1000 and orta_ust > 1000:
            if self.is_line_angled(angle):
                if angle < -self.angle_correction_threshold:
                    return "SAGA DON (AÇI DÜZELTMESİ)"  # Çizgi sağa eğik, sola dön
                elif angle > self.angle_correction_threshold:
                    return "SOLA DON (AÇI DÜZELTMESİ)"  # Çizgi sola eğik, sağa dön
            else:
                return "DUZ GIT"  # Çizgi düz, normal ilerle

        # MEVCUT ÜST BÖLGE KONTROLÜ
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
        # Bölge çizgileri
        cv2.line(frame, (self.region_width, 0), (self.region_width, self.height), (0, 255, 0), 2)
        cv2.line(frame, (2 * self.region_width, 0), (2 * self.region_width, self.height), (0, 255, 0), 2)
        cv2.line(frame, (0, self.region_height), (self.width, self.region_height), (0, 255, 0), 2)

        # *** YENİ: Merkez çizgisi ***
        cv2.line(frame, (self.width // 2, 0), (self.width // 2, self.height), (255, 0, 0), 1)

        # Bölge etiketleri
        cv2.putText(frame, "SOL UST", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "ORTA UST", (self.region_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "SAG UST", (2 * self.region_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    1)

        cv2.putText(frame, "SOL ALT", (10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "ORTA ALT", (self.region_width + 10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        cv2.putText(frame, "SAG ALT", (2 * self.region_width + 10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

        return frame

    def draw_detected_line(self, frame, line, center, angle):
        """*** YENİ: Tespit edilen çizgiyi ve bilgilerini çiz ***"""
        if line is not None:
            x1, y1, x2, y2 = line
            # Ana çizgiyi kırmızı renkte çiz
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            if center is not None:
                # Merkez noktasını yeşil daire ile işaretle
                cv2.circle(frame, center, 5, (0, 255, 0), -1)

                # Merkez noktasından dikey çizgi çiz (referans için)
                cv2.line(frame, (center[0], center[1] - 20), (center[0], center[1] + 20), (0, 255, 0), 2)

        return frame

    def run(self):
        """Ana döngü"""
        print("*** GELİŞTİRİLMİŞ Çizgi takibi başlatılıyor ***")
        print("Çıkmak için 'q' tuşuna basın")
        print("Video hızını ayarlamak için 'SPACE' ile duraklat/devam et")

        paused = False

        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("Video bitti veya okuma hatası!")
                    break

            # *** YENİ: Çizgi açısını tespit et ***
            angle, line_center, detected_line = self.detect_line_angle(frame)

            # Çizgi pozisyonunu tespit et (6 bölge)
            regions, thresh = self.detect_line_position(frame)

            # Viraj var mı kontrol et
            is_viraj = self.viraj_tespiti(regions)

            # *** GELİŞTİRİLMİŞ: Açı bilgisi ile hareket kararı ver ***
            if is_viraj:
                hareket = self.viraj_fonksiyonu(regions)
                mod = "VIRAJ MODU"
            else:
                hareket = self.duz_cizgi_fonksiyonu(regions, angle, line_center)
                mod = "DUZ CIZGI MODU"

            # Görselleştirme
            frame = self.draw_regions(frame)
            frame = self.draw_detected_line(frame, detected_line, line_center, angle)

            # Bilgileri formatla
            sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions
            bolge_bilgisi = f"U:[{sol_ust}, {orta_ust}, {sag_ust}] A:[{sol_alt}, {orta_alt}, {sag_alt}]"

            # *** GELİŞTİRİLMİŞ: Açı bilgisini de göster ***
            aci_bilgisi = f"Açı: {angle:.1f}°" if angle is not None else "Açı: Tespit edilemedi"

            # Ekrana yazdır
            cv2.putText(frame, f"Mod: {mod}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Hareket: {hareket}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, aci_bilgisi, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.putText(frame, bolge_bilgisi, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            if paused:
                cv2.putText(frame, "DURAKLATILDI - SPACE ile devam et", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255), 2)

            # Görüntüleri göster
            cv2.imshow('Geliştirilmiş Cizgi Takibi', frame)
            cv2.imshow('Threshold', thresh)

            # Tuş kontrolü
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print("Duraklatıldı" if paused else "Devam ediyor")

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