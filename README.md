# YOLOv8 ile Nesne ve Kablo Tespiti

Bu proje, YOLOv8 modelini kullanarak bir videodaki şekilleri ve kabloları tespit eden, siyah alanları analiz eden ve bu analizlere göre yönlendirme yapan bir Python uygulamasıdır. Uygulama, OpenCV ile video işleme yaparak gerçek zamanlı olarak nesneleri tanımlar ve bu nesnelerin kamera merkezine olan mesafelerini hesaplar.

## Özellikler

- **Kablo Tespiti:** YOLOv8 modeli ile video akışında 'Kablo' etiketiyle tanımlanan nesneler tespit edilir.
- **Siyah Alan Tespiti:** Siyah renkli alanlar tespit edilip, bu alanların merkez noktası belirlenir.
- **Yönlendirme:** Siyah alanın merkez noktası ile kameranın merkezi arasındaki mesafeye göre, kullanıcıya sağa, sola veya düz gitme önerisi sunulur.
- **FPS Hesaplama:** Her video karesi için saniye başına işlem yapılan kare sayısı (FPS) hesaplanır ve ekranda görüntülenir.

## Gereksinimler

- Python 3.x
- OpenCV (`opencv-python` ve `opencv-python-headless`)
- NumPy
- YOLOv8 (Ultralytics)
