import cv2
import numpy as np
import time


class ImprovedSmartLineFollowingWithSearch:
    def __init__(self, video_path=None):
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
        else:
            self.cap = cv2.VideoCapture(0)

        # Video boyutları
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.region_width = self.width // 3
        self.region_height = int(self.height * 0.6)

        # Çizgi tespiti
        self.lower_threshold = 0
        self.upper_threshold = 50
        self.minimum_line_pixels = 300

        # Çizgi kaybı analizi
        self.line_lost = False
        self.line_lost_time = 0
        self.line_lost_threshold = 0.8

        # Son hareket ve bölge analizi
        self.last_action = "DUZ GIT"
        self.last_regions = [0, 0, 0, 0, 0, 0]
        self.previous_regions = [0, 0, 0, 0, 0, 0]

        # Arama modu
        self.search_mode = False
        self.search_direction = None
        self.search_step = 0
        self.search_start_time = 0
        self.max_search_time = 10.0

    def detect_line_position(self, frame):
        """6 bölge çizgi tespiti"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.upper_threshold, 255, cv2.THRESH_BINARY_INV)

        # Üst yarım
        upper_half = thresh[0:self.region_height, :]
        upper_regions = []
        for i in range(3):
            start_x = i * self.region_width
            end_x = (i + 1) * self.region_width
            region = upper_half[:, start_x:end_x]
            white_pixels = np.sum(region == 255)
            upper_regions.append(white_pixels)

        # Alt yarım
        lower_half = thresh[self.region_height:, :]
        lower_regions = []
        for i in range(3):
            start_x = i * self.region_width
            end_x = (i + 1) * self.region_width
            region = lower_half[:, start_x:end_x]
            white_pixels = np.sum(region == 255)
            lower_regions.append(white_pixels)

        all_regions = upper_regions + lower_regions
        return all_regions, thresh

    def is_line_detected(self, regions):
        """Çizgi var mı kontrol et"""
        return sum(regions) > self.minimum_line_pixels

    def viraj_tespiti(self, regions):
        """Viraj var mı kontrol et - GELİŞTİRİLMİŞ VERSİYON"""
        sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions

        # KILIT KURAL: Orta alt + Orta üst varsa → DÜZ GİT (viraj değil)
        if orta_alt > 1000 and orta_ust > 1000:
            return False

        # Viraj durumları (orta alt bölgede çizgi varsa):
        if orta_alt > 1000:
            if sag_alt > 1000:  # Orta alt + Sağ alt
                return True
            elif sol_alt > 1000:  # Orta alt + Sol alt
                return True

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
        """Düz çizgi için hareket kararı - İYİLEŞTİRİLMİŞ VERSİYON"""
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

    def intelligent_search_direction(self, current_regions, last_action):
        """Akıllı arama yönü belirleme - GELİŞTİRİLMİŞ VERSİYON"""
        sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = current_regions

        # Bölge analizi
        sol_total = sol_ust + sol_alt
        sag_total = sag_ust + sag_alt
        orta_total = orta_ust + orta_alt
        toplam_piksel = sum(current_regions)

        print(f"🎯 Bölge Analizi: Sol={sol_total}, Orta={orta_total}, Sağ={sag_total}")
        print(f"📊 Son Hareket: {last_action}")

        # 1. DÜZ ÇİZGİ KAYBI DURUMU
        if last_action in ["DUZ GIT", "BEKLEMEDE"] and toplam_piksel < 300:
            print("🚨 DÜZ ÇİZGİ KAYBI TESPİT EDİLDİ!")

            # Hangi bölgede son çizgi kalıntısı var?
            if orta_total > sol_total and orta_total > sag_total:
                return "ORTA_ARAMA", "Düz çizgide orta bölgede kayıp"
            elif sol_total > sag_total:
                return "SOL_ARAMA", "Düz çizgide sol bölgede kayıp"
            else:
                return "SAG_ARAMA", "Düz çizgide sağ bölgede kayıp"

        # 2. YENGEÇ HAREKETİ SONRASI KAYIP
        if last_action in ["SOL YENGEC", "SAG YENGEC"] and toplam_piksel < 300:
            print("🦀 YENGEÇ HAREKETİ SONRASI KAYIP!")

            # Yengeç yönüne göre arama
            if last_action == "SOL YENGEC":
                return "SOL_ARAMA", "Sol yengeç sonrası kayıp"
            else:
                return "SAG_ARAMA", "Sağ yengeç sonrası kayıp"

        # 3. ÜST BÖLGE KONTROLÜ - YENİ EKLENEN MANTIK
        if sol_ust > 1000 and orta_ust <= 1000 and sag_ust <= 1000:
            print("🔍 ÜST SOL BÖLGEDE ÇİZGİ VAR!")
            return "SOL_ARAMA", "Üst sol bölgede çizgi mevcut"

        elif sag_ust > 1000 and orta_ust <= 1000 and sol_ust <= 1000:
            print("🔍 ÜST SAĞ BÖLGEDE ÇİZGİ VAR!")
            return "SAG_ARAMA", "Üst sağ bölgede çizgi mevcut"

        elif orta_ust > 1000 and sol_ust <= 1000 and sag_ust <= 1000:
            print("🔍 ÜST ORTA BÖLGEDE ÇİZGİ VAR!")
            return "ORTA_ARAMA", "Üst orta bölgede çizgi mevcut"

        # 4. NORMAL BÖLGE ANALİZİ
        if sol_total > 500 and sag_total > 500:
            return "ORTA_ARAMA", "Her iki tarafta da çizgi mevcut"
        elif sol_total > sag_total:
            return "SOL_ARAMA", "Sol bölgede daha fazla çizgi"
        else:
            return "SAG_ARAMA", "Sağ bölgede daha fazla çizgi"

    def check_line_loss(self, regions):
        """Çizgi kaybı kontrolü ve arama moduna geçiş"""
        current_time = time.time()

        if not self.is_line_detected(regions):
            if not self.line_lost:
                self.line_lost_time = current_time
                self.line_lost = True
                print("⚠️ Çizgi kaybolmaya başladı...")

            elif current_time - self.line_lost_time > self.line_lost_threshold:
                if not self.search_mode:
                    self.start_intelligent_search(regions)
        else:
            if self.line_lost or self.search_mode:
                self.stop_search_mode()
            self.line_lost = False

    def start_intelligent_search(self, current_regions):
        """Akıllı arama modunu başlat"""
        self.search_mode = True
        self.search_start_time = time.time()
        self.search_step = 0

        # Geliştirilmiş arama yönü belirleme
        direction, reason = self.intelligent_search_direction(current_regions, self.last_action)
        self.search_direction = direction

        print(f"🔍 GELİŞTİRİLMİŞ AKILLI ARAMA BAŞLADI")
        print(f"📍 Yön: {direction}")
        print(f"🧠 Sebep: {reason}")
        print(f"🎬 Son hareket: {self.last_action}")

    def stop_search_mode(self):
        """Arama modunu durdur"""
        self.search_mode = False
        self.search_step = 0
        self.search_direction = None
        print("✅ GELİŞTİRİLMİŞ ARAMA BİTTİ - Çizgi bulundu!")

    def execute_search_movement(self):
        """Arama hareketi uygula - GELİŞTİRİLMİŞ VERSİYON"""
        current_time = time.time()

        # Zaman aşımı kontrolü
        if current_time - self.search_start_time > self.max_search_time:
            self.stop_search_mode()
            return "ZAMAN AŞIMI - DURDUR"

        self.search_step += 1

        # Arama yönlerine göre hareket
        if self.search_direction == "SOL_ARAMA":
            return "SOL GENIS ARAMA"
        elif self.search_direction == "SAG_ARAMA":
            return "SAG GENIS ARAMA"
        elif self.search_direction == "ORTA_ARAMA":
            # Orta arama için dönüşümlü hareket
            if self.search_step % 4 < 2:
                return "ORTA SOL ARAMA"
            else:
                return "ORTA SAG ARAMA"
        else:
            return "SOL GENIS ARAMA"

    def draw_regions_and_info(self, frame, action, regions):
        """Görsel bilgileri çiz - GELİŞTİRİLMİŞ VERSİYON"""
        # Bölge çizgileri
        cv2.line(frame, (self.region_width, 0), (self.region_width, self.height), (0, 255, 0), 2)
        cv2.line(frame, (2 * self.region_width, 0), (2 * self.region_width, self.height), (0, 255, 0), 2)
        cv2.line(frame, (0, self.region_height), (self.width, self.region_height), (0, 255, 0), 2)

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

        # Durum bilgileri
        if self.search_mode:
            mod = f"🔍 GELISTIRILMIS ARAMA: {self.search_direction}"
            mod_color = (0, 0, 255)
        else:
            mod = "✅ GELISTIRILMIS TAKIP"
            mod_color = (0, 255, 0)

        cv2.putText(frame, mod, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mod_color, 2)
        cv2.putText(frame, f"Hareket: {action}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Bölge değerleri
        sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions
        bolge_bilgisi = f"U:[{sol_ust}, {orta_ust}, {sag_ust}] A:[{sol_alt}, {orta_alt}, {sag_alt}]"
        cv2.putText(frame, bolge_bilgisi, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if self.search_mode:
            cv2.putText(frame, f"Arama Adım: {self.search_step}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255),
                        1)
            remaining = self.max_search_time - (time.time() - self.search_start_time)
            cv2.putText(frame, f"Kalan: {remaining:.1f}s", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        return frame

    def run(self):
        """Ana döngü"""
        print("🤖 GELİŞTİRİLMİŞ Akıllı Çizgi Takip ve Arama Algoritması")
        print("📋 YENİ ÖZELLİKLER:")
        print("   - Üst bölge kontrolü ile geliştirilmiş arama")
        print("   - Yengeç hareketi sonrası akıllı arama")
        print("   - Orta arama için dönüşümlü hareket")
        print("🎮 Çıkmak için 'q', Duraklat için 'SPACE'")

        paused = False

        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("Video bitti!")
                    break

            # Çizgi tespiti
            regions, thresh = self.detect_line_position(frame)

            # Çizgi kaybı kontrolü
            self.check_line_loss(regions)

            # Hareket kararı
            if self.search_mode:
                hareket = self.execute_search_movement()
            else:
                if self.is_line_detected(regions):
                    is_viraj = self.viraj_tespiti(regions)
                    if is_viraj:
                        hareket = self.viraj_fonksiyonu(regions)
                    else:
                        hareket = self.duz_cizgi_fonksiyonu(regions)
                else:
                    hareket = "BEKLEMEDE"

            # Son durumu kaydet
            self.last_action = hareket
            self.previous_regions = self.last_regions.copy()
            self.last_regions = regions.copy()

            # Görselleştirme
            frame = self.draw_regions_and_info(frame, hareket, regions)

            if paused:
                cv2.putText(frame, "⏸️ DURAKLATILDI", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow('Geliştirilmiş Akıllı Çizgi Takip', frame)
            cv2.imshow('Threshold', thresh)

            # Kontroller
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused

        self.cap.release()
        cv2.destroyAllWindows()


# Kullanım
if __name__ == "__main__":
    try:
        video_path = "C:/Users/user/Downloads/video3.mp4"
        algorithm = ImprovedSmartLineFollowingWithSearch(video_path)
        algorithm.run()
    except Exception as e:
        print(f"Hata: {e}")