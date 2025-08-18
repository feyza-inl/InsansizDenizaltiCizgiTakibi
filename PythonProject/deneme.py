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

        # Bölge ayarları
        self.region_width = self.process_width // 3
        self.region_height = int(self.height * 0.5 / self.scale_factor)

        # Çizgi parametreleri
        self.lower_threshold = 0
        self.upper_threshold = 50
        self.max_allowed_angle = 20
        self.min_line_length = 30
        self.angle_correction_threshold = 10

        # FPS
        self.prev_time = time.time()
        self.fps = 0

        # Arama parametreleri
        self.son_cizgi_yonu = "ORTA"
        self.cizgi_kayip_sayaci = 0
        self.kayip_esigi = 3
        self.minimum_pixel_esigi = 500

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
            return "SOL ARAMA"
        elif self.son_cizgi_yonu == "SAG":
            print("🔄 Sag arama")
            return "SAG ARAMA"
        else:
            print("🔄 Orta arama")
            return "ORTA ARAMA"

    def preprocess_frame(self, frame):
        try:
            small_frame = cv2.resize(frame, (self.process_width, int(self.height / self.scale_factor)))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, self.upper_threshold, 255, cv2.THRESH_BINARY_INV)
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            return gray, thresh, small_frame
        except:
            empty = np.zeros((int(self.height / self.scale_factor), self.process_width), dtype=np.uint8)
            return empty, empty, empty

    def detect_line_angle(self, gray_frame):
        try:
            lines = cv2.HoughLinesP(gray_frame, 1, np.pi / 180,
                                    threshold=30,
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

        if orta_alt > 1000 and orta_ust > 1000:
            return False

        # YENİ MANTIK: Hem üst hem alt varsa yengeç, sadece alt varsa viraj

        # Sağ taraf kontrolü
        if sag_ust > 1000 and sag_alt > 1000:
            # Hem sağ üst hem sağ alt varsa -> SAG YENGEC (viraj değil)
            return False
        elif sag_alt > 1000 and orta_alt <= 1000 and sol_alt <= 1000:
            # Sadece sağ alt varsa -> SAG VIRAJ
            return True

        # Sol taraf kontrolü
        if sol_ust > 1000 and sol_alt > 1000:
            # Hem sol üst hem sol alt varsa -> SOL YENGEC (viraj değil)
            return False
        elif sol_alt > 1000 and orta_alt <= 1000 and sag_alt <= 1000:
            # Sadece sol alt varsa -> SOL VIRAJ
            return True

        if orta_alt > 1000:
            if sag_alt > 1000 or sol_alt > 1000:
                return True

        return False

    def viraj_fonksiyonu(self, regions):
        sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions

        if orta_alt > 1000 and sag_alt > 1000:
            print("🔄 saga don")
            return "SAGA DON"
        elif orta_alt > 1000 and sol_alt > 1000:
            print("🔄 Sol don")
            return "SOLA DON"
        # YENİ: Sadece sağ alt veya sadece sol alt için viraj
        elif sag_alt > 1000 and orta_alt <= 1000 and sol_alt <= 1000:
            print("🔄 saga don")
            return "SAGA DON"
        elif sol_alt > 1000 and orta_alt <= 1000 and sag_alt <= 1000:
            print("🔄 Sol don")
            return "SOLA DON"
        else:
            return "VIRAJ TESPIT EDILEMEDI"

    def duz_cizgi_fonksiyonu(self, regions, angle=None, line_center=None):
        sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions

        if orta_alt > 1000 and orta_ust > 1000:
            if self.is_line_angled(angle):
                if angle < -self.angle_correction_threshold:
                    print("🔄 saga don")
                    return "SAGA DON (AÇI DÜZELTMESİ)"
                elif angle > self.angle_correction_threshold:
                    print("🔄 Sol don")
                    return "SOLA DON (AÇI DÜZELTMESİ)"
            else:
                print("🔄 duz git")
                return "DUZ GIT"

        # YENİ MANTIK: Hem üst hem alt varsa yengeç hareketi
        if sag_ust > 1000 and sag_alt > 1000:
            print("🔄 sag yengec")
            return "SAG YENGEC"
        elif sol_ust > 1000 and sol_alt > 1000:
            print("🔄 Sol yengec")
            return "SOL YENGEC"
        # Sadece üst bölgeler için yengeç hareketi
        elif sol_ust > 1000 and orta_ust <= 1000 and sag_ust <= 1000:
            print("🔄 Sol yengec")
            return "SOL YENGEC"
        elif sag_ust > 1000 and orta_ust <= 1000 and sol_ust <= 1000:
            print("🔄 sag yengec")
            return "SAG YENGEC"
        elif orta_ust > 1000 and sol_ust <= 1000 and sag_ust <= 1000:
            print("🔄 duz git")
            return "DUZ GIT"

        # Sadece orta bölge varsa düz git
        if orta_alt > 1000:
            print("🔄 duz git")
            return "DUZ GIT"
        else:
            print("🔄 duz git")
            return "DUZ GIT"

    def draw_info(self, frame, mod, hareket, angle, regions, cizgi_mevcut):
        """Bilgileri çiz"""
        try:
            # Temel bilgiler
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (self.width - 100, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"Mod: {mod}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Hareket: {hareket}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Açı bilgisi
            aci_bilgisi = f"Açı: {angle:.1f}°" if angle is not None else "Açı: --"
            cv2.putText(frame, aci_bilgisi, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            # Çizgi durumu
            cv2.putText(frame, f"Cizgi: {'VAR' if cizgi_mevcut else 'YOK'}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if cizgi_mevcut else (0, 0, 255), 1)

            # Bölgeler - kompakt
            sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions
            region_text = f"U:[{sol_ust},{orta_ust},{sag_ust}] A:[{sol_alt},{orta_alt},{sag_alt}]"
            cv2.putText(frame, region_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Son yön bilgisi
            cv2.putText(frame, f"Son Yön: {self.son_cizgi_yonu} | Kayıp: {self.cizgi_kayip_sayaci}",
                        (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Bölge çizgileri
            rw = int(self.region_width * self.scale_factor)
            rh = int(self.region_height * self.scale_factor)
            cv2.line(frame, (rw, 0), (rw, self.height), (0, 255, 0), 1)
            cv2.line(frame, (rw * 2, 0), (rw * 2, self.height), (0, 255, 0), 1)
            cv2.line(frame, (0, rh), (self.width, rh), (0, 255, 0), 1)

            # Merkez çizgisi
            cv2.line(frame, (self.width // 2, 0), (self.width // 2, self.height), (255, 0, 0), 1)

            # Bölge etiketleri
            cv2.putText(frame, "SOL UST", (10, rh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "ORTA UST", (rw + 10, rh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "SAG UST", (rw * 2 + 10, rh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.putText(frame, "SOL ALT", (10, self.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "ORTA ALT", (rw + 10, self.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                        1)
            cv2.putText(frame, "SAG ALT", (rw * 2 + 10, self.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)

            # Mod çerçevesi
            if mod == "ARAMA MODU":
                cv2.rectangle(frame, (2, 2), (self.width - 2, self.height - 2), (0, 0, 255), 3)
            elif mod == "VIRAJ MODU":
                cv2.rectangle(frame, (2, 2), (self.width - 2, self.height - 2), (255, 0, 0), 3)
            elif "YENGEC" in hareket:
                cv2.rectangle(frame, (2, 2), (self.width - 2, self.height - 2), (0, 255, 255), 3)

        except:
            pass

    def run(self):
        """Ana döngü"""
        print("*** ÇİZGİ TAKİBİ - VIDEO TEST VERSİYONU ***")
        print("Çıkmak için 'q' tuşuna basın")
        print("Duraklatmak için 'SPACE' tuşuna basın")
        print("Sıfırlamak için 'r' tuşuna basın")

        paused = False
        frame_count = 0
        last_fps_time = time.time()

        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("Video bitti veya okuma hatası!")
                    break

                frame_count += 1
                current_time = time.time()

                # FPS hesapla (her 30 frame'de)
                if frame_count % 30 == 0:
                    if current_time - last_fps_time > 0:
                        self.fps = 30 / (current_time - last_fps_time)
                    last_fps_time = current_time

                # Görüntü işleme
                gray, thresh, small_frame = self.preprocess_frame(frame)
                angle, center, lines = self.detect_line_angle(thresh)
                regions, _ = self.detect_line_position(thresh)

                # Çizgi kontrolü
                cizgi_mevcut = self.cizgi_var_mi(regions)

                # Hareket kararı
                if cizgi_mevcut:
                    self.cizgi_kayip_sayaci = 0
                    self.son_cizgi_yonunu_guncelle(regions)

                    is_viraj = self.viraj_tespiti(regions)

                    if is_viraj:
                        hareket = self.viraj_fonksiyonu(regions)
                        mod = "VIRAJ MODU"
                    else:
                        hareket = self.duz_cizgi_fonksiyonu(regions, angle, center)
                        mod = "DUZ CIZGI MODU"
                else:
                    self.cizgi_kayip_sayaci += 1

                    if self.cizgi_kayip_sayaci >= self.kayip_esigi:
                        hareket = self.arama_modu_karar()
                        mod = "ARAMA MODU"
                    else:
                        hareket = "BEKLE"
                        mod = "BEKLE MODU"

                # Görselleştirme
                self.draw_info(frame, mod, hareket, angle, regions, cizgi_mevcut)

                # Tespit edilen çizgileri çiz
                if lines:
                    for line in lines:
                        x1, y1, x2, y2 = [int(x * self.scale_factor) for x in line]
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Görüntüleri göster
                cv2.imshow('Cizgi Takibi', frame)

                # Threshold küçük boyutta
                if thresh is not None:
                    small_thresh = cv2.resize(thresh, (320, 240))
                    cv2.imshow('Threshold', small_thresh)

                # Konsol çıktısı (her 60 frame'de)
                if frame_count % 60 == 0:
                    print(f"📊 Frame: {frame_count}, FPS: {self.fps:.1f}, Mod: {mod}, Hareket: {hareket}")

            # Tuş kontrolü
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Q tuşu ile çıkış")
                break
            elif key == ord(' '):
                paused = not paused
                print("⏸ Duraklatıldı" if paused else "▶ Devam ediyor")
            elif key == ord('r'):
                # Sistem sıfırla
                self.cizgi_kayip_sayaci = 0
                self.son_cizgi_yonu = "ORTA"
                print("🔄 Sistem sıfırlandı")

        # Temizlik
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        # Video dosyası yolu - buraya kendi video dosyanızın yolunu yazın
        video_path = "C:/Users/user/Downloads/video7.mp4"  # 0 = webcam, video dosyası için path verin

        algorithm = LineFollowingAlgorithm(video_path)
        algorithm.run()
    except KeyboardInterrupt:
        print("\nProgram durduruldu.")
    except Exception as e:
        print(f"Hata: {e}")
        print("Video dosyası yolunu kontrol edin.")