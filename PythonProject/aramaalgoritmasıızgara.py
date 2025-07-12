import cv2
import numpy as np


class LineFollowingAlgorithm:
    def __init__(self, video_path=None):
        if video_path:
            self.cap = cv2.VideoCapture(video_path)  # Video dosyasından okuma
        else:
            self.cap = cv2.VideoCapture(0)  # Kamera başlatma (0 = varsayılan kamera)

        # Video boyutlarını al
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Ekranı 6 bölgeye ayırma (3 dikey x 2 yatay)
        self.region_width = self.width // 3
        self.region_height = int(self.height * 0.6)  # Yatay çizgiyi %70'e indirdim (daha aşağı)

        # Çizgi rengi eşiği (siyah çizgi için)
        self.lower_threshold = 0
        self.upper_threshold = 50

        # YENİ EKLENEN ÖZELLİKLER - ARAMA MODU İÇİN
        self.son_cizgi_yonu = "ORTA"  # Son görülen çizginin yönü
        self.cizgi_kayip_sayaci = 0  # Çizgi kaybı sayacı
        self.kayip_esigi = 3  # Kaç frame çizgi görülmezse arama moduna geçsin
        self.minimum_pixel_esigi = 500  # Çizgi var sayılması için minimum pixel sayısı

    def detect_line_position(self, frame):
        """Çizginin hangi bölgede olduğunu tespit eder - 6 bölge"""
        # Görüntüyü gri tonlamaya çevir
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Çizgi tespiti için threshold uygula
        _, thresh = cv2.threshold(gray, self.upper_threshold, 255, cv2.THRESH_BINARY_INV)

        # ÜST YARIM (0 - region_height arası)
        upper_half = thresh[0:self.region_height, :]

        # ALT YARIM (region_height - height arası)
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

        # 6 bölgelik sonuç: [sol_üst, orta_üst, sağ_üst, sol_alt, orta_alt, sağ_alt]
        all_regions = upper_regions + lower_regions

        return all_regions, thresh

    def cizgi_var_mi(self, regions):
        """Çizgi var mı yok mu kontrol et - YENİ FONKSİYON"""
        # Herhangi bir bölgede minimum eşik değerinin üzerinde pixel var mı?
        return any(region > self.minimum_pixel_esigi for region in regions)

    def son_cizgi_yonunu_guncelle(self, regions):
        """Son görülen çizginin yönünü güncelle - YENİ FONKSİYON"""
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

    def arama_modu_karar(self):
        """Arama modunda hangi yöne gidileceğini belirle - YENİ FONKSİYON"""
        if self.son_cizgi_yonu == "SOL":
            return "SOL ARAMA"  # Sola doğru salyangoz hareketi
        elif self.son_cizgi_yonu == "SAG":
            return "SAG ARAMA"  # Sağa doğru salyangoz hareketi
        else:
            return "ORTA ARAMA"  # Orta bölgede kaybolmuşsa hafif zigzag

    def viraj_tespiti(self, regions):
        """Viraj var mı yok mu kontrol et - 6 bölge mantığı"""
        sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions

        # KILIT KURAL: Orta alt + Orta üst varsa → DÜZ GİT (viraj değil)
        if orta_alt > 1000 and orta_ust > 1000:
            return False  # Viraj değil

        # Viraj durumları:
        # 1. Orta alt + Sağ alt = SAĞA DÖN
        # 2. Orta alt + Sol alt = SOLA DÖN

        if orta_alt > 1000:
            if sag_alt > 1000:  # Orta alt + Sağ alt
                return True  # Sağa dön virajı
            elif sol_alt > 1000:  # Orta alt + Sol alt
                return True  # Sola dön virajı

        return False

    def viraj_fonksiyonu(self, regions):
        """Viraj tespiti ve hareket kararı"""
        sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions

        # Orta alt + Sağ alt = SAĞA DÖN
        if orta_alt > 1000 and sag_alt > 1000:
            return "SAGA DON"

        # Orta alt + Sol alt = SOLA DÖN
        elif orta_alt > 1000 and sol_alt > 1000:
            return "SOLA DON"

        else:
            return "VIRAJ TESPIT EDILEMEDI"

    def duz_cizgi_fonksiyonu(self, regions):
        """Düz çizgi için hareket kararı"""
        sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions

        # YENİ EKLENEN KONTROLLER - ÜST BÖLGE KONTROLÜ
        # Eğer çizgi sadece sol üstte varsa sol yengeç yap
        if sol_ust > 1000 and orta_ust <= 1000 and sag_ust <= 1000:
            return "SOL YENGEC"

        # Eğer çizgi sadece sağ üstte varsa sağ yengeç yap
        elif sag_ust > 1000 and orta_ust <= 1000 and sol_ust <= 1000:
            return "SAG YENGEC"

        # Eğer çizgi sadece orta üstte varsa düz git
        elif orta_ust > 1000 and sol_ust <= 1000 and sag_ust <= 1000:
            return "DUZ GIT"

        # MEVCUT MANTIK - HİÇ DEĞİŞTİRİLMEDİ
        # Orta alt + Orta üst = DÜZ GİT
        if orta_alt > 1000 and orta_ust > 1000:
            return "DUZ GIT"

        # Sadece alt bölgelere bak (eski mantık)
        lower_regions = [sol_alt, orta_alt, sag_alt]
        max_index = np.argmax(lower_regions)

        if max_index == 0:  # Sol alt
            return "SOL YENGEC"
        elif max_index == 1:  # Orta alt
            return "DUZ GIT"
        else:  # Sağ alt
            return "SAG YENGEC"

    def draw_regions(self, frame):
        """Bölgeleri görsel olarak çiz"""
        # Dikey çizgiler
        cv2.line(frame, (self.region_width, 0), (self.region_width, self.height), (0, 255, 0), 2)
        cv2.line(frame, (2 * self.region_width, 0), (2 * self.region_width, self.height), (0, 255, 0), 2)

        # Yatay çizgi
        cv2.line(frame, (0, self.region_height), (self.width, self.region_height), (0, 255, 0), 2)

        # Üst bölge etiketleri
        cv2.putText(frame, "SOL UST", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "ORTA UST", (self.region_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "SAG UST", (2 * self.region_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    1)

        # Alt bölge etiketleri
        cv2.putText(frame, "SOL ALT", (10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "ORTA ALT", (self.region_width + 10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        cv2.putText(frame, "SAG ALT", (2 * self.region_width + 10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

        return frame

    def run(self):
        """Ana döngü - GELİŞTİRİLMİŞ VERSİYON"""
        print("Çizgi takibi başlatılıyor...")
        print("Çıkmak için 'q' tuşuna basın")
        print("Video hızını ayarlamak için 'SPACE' ile duraklat/devam et")

        paused = False

        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("Video bitti veya okuma hatası!")
                    break

            # Çizgi pozisyonunu tespit et (6 bölge)
            regions, thresh = self.detect_line_position(frame)

            # Çizgi var mı kontrol et
            cizgi_mevcut = self.cizgi_var_mi(regions)

            if cizgi_mevcut:
                # Çizgi varsa normal işlem
                self.cizgi_kayip_sayaci = 0  # Sayacı sıfırla
                self.son_cizgi_yonunu_guncelle(regions)  # Son yönü güncelle

                # Viraj var mı kontrol et
                is_viraj = self.viraj_tespiti(regions)

                # Hareket kararı ver
                if is_viraj:
                    hareket = self.viraj_fonksiyonu(regions)
                    mod = "VIRAJ MODU"
                else:
                    hareket = self.duz_cizgi_fonksiyonu(regions)
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

            # Bölgeleri çiz
            frame = self.draw_regions(frame)

            # Bölge değerlerini formatla
            sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions
            bolge_bilgisi = f"U:[{sol_ust}, {orta_ust}, {sag_ust}] A:[{sol_alt}, {orta_alt}, {sag_alt}]"

            # Bilgileri ekrana yazdır
            cv2.putText(frame, f"Mod: {mod}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Hareket: {hareket}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, bolge_bilgisi, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # YENİ BİLGİLER
            cv2.putText(frame, f"Son Cizgi Yonu: {self.son_cizgi_yonu}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)
            cv2.putText(frame, f"Kayip Sayaci: {self.cizgi_kayip_sayaci}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)
            cv2.putText(frame, f"Cizgi Mevcut: {'EVET' if cizgi_mevcut else 'HAYIR'}", (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if cizgi_mevcut else (0, 0, 255), 1)

            if paused:
                cv2.putText(frame, "DURAKLATILDI - SPACE ile devam et", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255), 2)

            # Arama modundaysa kırmızı çerçeve çiz
            if mod == "ARAMA MODU":
                cv2.rectangle(frame, (5, 5), (self.width - 5, self.height - 5), (0, 0, 255), 5)

            # Görüntüleri göster
            cv2.imshow('Cizgi Takibi', frame)
            cv2.imshow('Threshold', thresh)

            # Tuş kontrolü
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # SPACE tuşu
                paused = not paused
                print("Duraklatıldı" if paused else "Devam ediyor")
            elif key == ord('r'):  # R tuşu - Reset
                self.cizgi_kayip_sayaci = 0
                self.son_cizgi_yonu = "ORTA"
                print("Arama modu sıfırlandı")

        # Temizlik
        self.cap.release()
        cv2.destroyAllWindows()


# Video ile kullanım
if __name__ == "__main__":
    try:
        # Video dosyası yolunu buraya yazın
        #video_path = "C:/Users/user/Downloads/video6.mp4"  # Kendi video dosyanızın yolunu yazın
        video_path = 0 # Kamera için None yapın

        # Kamera için None bırakın, video için dosya yolunu verin
        algorithm = LineFollowingAlgorithm(video_path)  # Video için
        # algorithm = LineFollowingAlgorithm()  # Kamera için

        algorithm.run()
    except KeyboardInterrupt:
        print("\nProgram durduruldu.")
    except Exception as e:
        print(f"Hata: {e}")
        print("Video dosyası yolunu kontrol edin.")