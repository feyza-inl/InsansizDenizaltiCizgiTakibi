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
        self.angle_correction_threshold = 15  # pasted2'deki değer

        # FPS
        self.prev_time = time.time()
        self.fps = 0

        # Arama parametreleri
        self.son_cizgi_yonu = "ORTA"
        self.cizgi_kayip_sayaci = 0
        self.kayip_esigi = 3
        self.minimum_pixel_esigi = 500

        # YENİ: K-means parametreleri
        self.kmeans_clusters = 3  # Siyah çizgi, su, diğer objeler
        self.use_kmeans = True  # K-means'i aktif/pasif yapma
        self.kmeans_counter = 0  # Her frame K-means yapmamak için sayaç
        self.kmeans_interval = 10  # Her 10 frame'de bir K-means çalıştır
        self.last_kmeans_mask = None  # Son K-means sonucunu sakla

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
        # PASTED2'DEN ALINAN ORİJİNAL ALGORİTMA MANTIĞI
        if self.son_cizgi_yonu == "SOL":
            print("🔄 sol arama")
            return "SOL ARAMA"
        elif self.son_cizgi_yonu == "SAG":
            print("🔄 Sag arama")
            return "SAG ARAMA"
        else:
            print("🔄 Orta arama")
            return "ORTA ARAMA"

    # YENİ: Piksel yoğunluğu hesaplama fonksiyonu
    def calculate_pixel_density(self, image):
        """Piksel yoğunluğunu hesapla"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

            # Gradyan yoğunluğu hesapla
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

            # Normalleştir
            gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)

            return gradient_magnitude.astype(np.uint8)
        except:
            return np.zeros_like(image[:, :, 0] if len(image.shape) == 3 else image)

    # YENİ: Hızlı K-means implementasyonu (çok daha hızlı)
    def fast_kmeans(self, features, k=3, max_iters=3):
        """Hızlı K-means implementasyonu"""
        try:
            n_samples, n_features = features.shape

            # Sadece her 10. pikseli kullan (çok daha hızlı)
            sample_indices = np.arange(0, n_samples, 10)
            sample_features = features[sample_indices]

            # Rastgele merkezler başlat
            np.random.seed(42)
            centroids = sample_features[np.random.choice(len(sample_features), k, replace=False)]

            for _ in range(max_iters):  # Sadece 3 iterasyon
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

    # YENİ: Hızlı K-means segmentasyon fonksiyonu
    def segment_with_kmeans(self, image, pixel_density):
        """Hızlı K-means ile segmentasyon yap"""
        try:
            h, w = image.shape[:2]

            # Görüntüyü daha da küçült (çok daha hızlı)
            small_h, small_w = h // 2, w // 2
            small_image = cv2.resize(image, (small_w, small_h))
            small_density = cv2.resize(pixel_density, (small_w, small_h))

            # Özellik vektörü oluştur (sadece küçük görüntüden)
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
            # Hata durumunda normal threshold döndür
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            _, thresh = cv2.threshold(gray, self.upper_threshold, 255, cv2.THRESH_BINARY_INV)
            return thresh // 255, None

    # YENİ: Çizgi kümesini belirleme fonksiyonu (çok daha seçici)
    def identify_line_cluster(self, image, segmented, centroids):
        """Siyah çizgiye ait kümeyi belirle - ÇOK SEÇİCİ"""
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
                    # Ortalama renk değeri
                    mean_color = np.mean(cluster_pixels, axis=0)
                    mean_brightness = np.mean(mean_color)

                    # Piksel sayısı
                    pixel_count = np.sum(mask)

                    # ÇİZGİ KRİTERLERİ - ÇOK SIKI:
                    # 1. Çok karanlık olmalı (< 80)
                    is_dark_enough = mean_brightness < 80

                    # 2. Çok fazla piksel olmamalı (kayalık değil, çizgi)
                    is_reasonable_size = 100 < pixel_count < (h * w * 0.3)

                    # 3. Şekil kontrolü - çizgi gibi uzun ve ince olmalı
                    y_coords, x_coords = np.where(mask)
                    if len(y_coords) > 0:
                        # Çizginin genişlik/yükseklik oranı
                        width_span = np.max(x_coords) - np.min(x_coords)
                        height_span = np.max(y_coords) - np.min(y_coords)
                        aspect_ratio = max(width_span, height_span) / (min(width_span, height_span) + 1)

                        # Çizgi uzun ve ince olmalı
                        is_line_shaped = aspect_ratio > 3

                        # Çizgi merkeze yakın olmalı
                        center_x = np.mean(x_coords)
                        distance_from_center = abs(center_x - w / 2) / (w / 2)
                        is_center_aligned = distance_from_center < 0.7
                    else:
                        is_line_shaped = False
                        is_center_aligned = False

                    # 4. Renk tutarlılığı - gerçek siyah çizgi uniform olmalı
                    color_std = np.std(cluster_pixels, axis=0)
                    is_uniform = np.mean(color_std) < 30

                    # TOPLAM SKOR - Tüm kriterler sağlanmalı
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
                        'pixel_count': pixel_count,
                        'is_dark': is_dark_enough,
                        'is_size_ok': is_reasonable_size,
                        'is_line_shaped': is_line_shaped,
                        'is_centered': is_center_aligned,
                        'is_uniform': is_uniform
                    })

            # En yüksek skoru alan ve minimum kriterleri sağlayan kümeyi seç
            valid_clusters = [c for c in cluster_stats if c['score'] >= 70]  # Minimum 70 puan

            if valid_clusters:
                best_cluster = max(valid_clusters, key=lambda x: x['score'])
                return best_cluster['id']
            else:
                # Hiç uygun küme yoksa, en karanlık olanı al ama uyar
                if cluster_stats:
                    darkest = min(cluster_stats, key=lambda x: x['brightness'])
                    print(f"⚠️ Çizgi kriterleri sağlanamadı, en karanlık küme seçildi: {darkest['id']}")
                    return darkest['id']
                return 1

        except Exception as e:
            print(f"Çizgi tespit hatası: {e}")
            return 1

    # YENİ: Gelişmiş K-means maskesi çıkarma
    def extract_kmeans_mask(self, segmented, line_cluster_id):
        """K-means sonucundan çizgi maskesini çıkar - GELİŞMİŞ FİLTRELEME"""
        try:
            line_mask = (segmented == line_cluster_id).astype(np.uint8) * 255

            # 1. Küçük gürültüleri temizle
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, kernel_small)

            # 2. Kontür analizi ile sadece çizgi benzeri şekilleri al
            contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Yeni temiz maske oluştur
            clean_mask = np.zeros_like(line_mask)

            for contour in contours:
                area = cv2.contourArea(contour)

                # Kontür çok küçük veya çok büyükse atla
                if area < 100 or area > line_mask.shape[0] * line_mask.shape[1] * 0.4:
                    continue

                # Bounding rectangle hesapla
                x, y, w, h = cv2.boundingRect(contour)

                # Çizgi kriterleri:
                aspect_ratio = max(w, h) / (min(w, h) + 1)

                # En az 2:1 oranında uzun olmalı (çizgi gibi)
                if aspect_ratio > 2:
                    # Bu konturu çizgi olarak kabul et
                    cv2.drawContours(clean_mask, [contour], -1, 255, -1)

            # 3. Son temizlik
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)

            return clean_mask

        except Exception as e:
            print(f"Maske çıkarma hatası: {e}")
            return np.zeros(segmented.shape, dtype=np.uint8)

    def preprocess_frame(self, frame):
        try:
            small_frame = cv2.resize(frame, (self.process_width, int(self.height / self.scale_factor)))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            # YENİ: K-means kullanımı (her 10 frame'de bir)
            if self.use_kmeans and self.kmeans_counter % self.kmeans_interval == 0:
                try:
                    # Piksel yoğunluğunu hesapla
                    pixel_density = self.calculate_pixel_density(small_frame)

                    # K-means segmentasyon
                    segmented, centroids = self.segment_with_kmeans(small_frame, pixel_density)

                    # Çizgi kümesini belirle
                    line_cluster_id = self.identify_line_cluster(small_frame, segmented, centroids)

                    # K-means maskesini çıkar
                    kmeans_mask = self.extract_kmeans_mask(segmented, line_cluster_id)

                    # Başarılı K-means sonucunu sakla - DAHA SIKI KONTROL
                    if np.sum(kmeans_mask) > 200 and np.sum(kmeans_mask) < (
                            small_frame.shape[0] * small_frame.shape[1] * 0.5):
                        self.last_kmeans_mask = kmeans_mask
                        self.kmeans_counter += 1
                        return gray, kmeans_mask, small_frame
                except Exception as e:
                    print(f"K-means işleminde hata: {e}")

            # Eğer yakın zamanda K-means çalıştıysa ve sonuç varsa onu kullan
            elif self.use_kmeans and self.last_kmeans_mask is not None and self.kmeans_counter % self.kmeans_interval < 5:
                self.kmeans_counter += 1
                return gray, self.last_kmeans_mask, small_frame

            # Normal threshold (K-means başarısızsa veya devre dışıysa)
            _, thresh = cv2.threshold(gray, self.upper_threshold, 255, cv2.THRESH_BINARY_INV)
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            self.kmeans_counter += 1
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
        # PASTED2'DEN ALINAN ORİJİNAL ALGORİTMA MANTIĞI
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
        # PASTED2'DEN ALINAN ORİJİNAL ALGORİTMA MANTIĞI
        sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions

        if orta_alt > 1000 and sag_alt > 1000:
            print("🔄 saga don ")
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
        # PASTED2'DEN ALINAN ORİJİNAL ALGORİTMA MANTIĞI
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

            # YENİ: K-means durumu ve interval bilgisi
            kmeans_status = f"AKTIF ({self.kmeans_counter % self.kmeans_interval}/{self.kmeans_interval})" if self.use_kmeans else "PASIF"
            cv2.putText(frame, f"K-means: {kmeans_status}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0) if self.use_kmeans else (0, 0, 255), 1)

            # Bölgeler - kompakt
            sol_ust, orta_ust, sag_ust, sol_alt, orta_alt, sag_alt = regions
            region_text = f"U:[{sol_ust},{orta_ust},{sag_ust}] A:[{sol_alt},{orta_alt},{sag_alt}]"
            cv2.putText(frame, region_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Son yön bilgisi
            cv2.putText(frame, f"Son Yön: {self.son_cizgi_yonu} | Kayıp: {self.cizgi_kayip_sayaci}",
                        (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

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
        print("*** ÇİZGİ TAKİBİ - PASTED2 ALGORİTMA MANTIĞI + K-MEANS ***")
        print("Çıkmak için 'q' tuşuna basın")
        print("Duraklatmak için 'SPACE' tuşuna basın")
        print("Sıfırlamak için 'r' tuşuna basın")
        print("K-means açma/kapama için 'k' tuşuna basın")
        print("K-means hızı için 'i' tuşuna basın (5/10 frame interval)")

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

                # PASTED2'DEN ALINAN ORİJİNAL HAREKET KARARI MANTIĞI
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
                cv2.imshow('Cizgi Takibi - PASTED2 Algoritma', frame)

                # Threshold küçük boyutta
                if thresh is not None:
                    small_thresh = cv2.resize(thresh, (320, 240))
                    cv2.imshow('Threshold', small_thresh)

                # Konsol çıktısı (her 60 frame'de)
                if frame_count % 60 == 0:
                    kmeans_status = f"K-means AKTIF (her {self.kmeans_interval} frame)" if self.use_kmeans else "Normal Threshold"
                    print(
                        f"📊 Frame: {frame_count}, FPS: {self.fps:.1f}, Mod: {mod}, Hareket: {hareket}, Segmentasyon: {kmeans_status}")

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
            elif key == ord('k'):  # YENİ: K-means açma/kapama
                self.use_kmeans = not self.use_kmeans
                print(f"🔧 K-means {'AKTİF' if self.use_kmeans else 'PASİF'}")
            elif key == ord('i'):  # YENİ: K-means interval ayarı
                self.kmeans_interval = 5 if self.kmeans_interval == 10 else 10
                print(f"🔧 K-means interval: her {self.kmeans_interval} frame")

        # Temizlik
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        # Video dosyası yolu - buraya kendi video dosyanızın yolunu yazın
        video_path = r"C:\Users\user\Downloads\video1.mp4"  # 0 = webcam, video dosyası için path verin

        algorithm = LineFollowingAlgorithm(video_path)
        algorithm.run()
    except KeyboardInterrupt:
        print("\nProgram durduruldu.")
    except Exception as e:
        print(f"Hata: {e}")
        print("Video dosyası yolunu kontrol edin.")