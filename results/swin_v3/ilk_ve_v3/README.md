# İlk veri bölmesi ve V3 sonuçları

- [Genel doğruluk ve F1 karşılaştırması](01_genel_performans.png)
- [Sınıf yakalama oranları](02_sinif_yakalama.png)
- [Swin deneyleri ve glioma hataları](03_swin_deneyleri.png)
- [Karşılaştırma tablosu](karsilastirma.csv)

İlk test 1.600, V3 testi 1.579 görüntü içerir. CNN için ilk model ve güncel V3 eğitimi, Swin için ilk model ve V3 başlangıç ağırlığı karşılaştırıldı. Swin grafiği ayrıca kayıtlı yerel RTX 3050 ve Colab T4 ince ayar sonuçlarını gösterir. İlk testte 400, V3 testte 383 glioma vardır. Glioma → tümör yok yüzdesi bu paydalarla hesaplandı.

İlk ölçümler kayıtlı model performansı analizinden, güncel ölçümler V3 test çıkarımlarından alındı. Swin ince ayar değerleri [deney sonuçları](../README.md) ve [Swin/SAM grafiklerinin kaynak notları](../swin_sam_grafikleri/README.md) ile eşleşir.

Veri bölmesi ve model ağırlıkları birlikte değiştiğinden farklar veri temizliğinin tek başına etkisini ölçmez. Hasta düzeyinde bağımsızlık doğrulanamadı.
