# Swin Aşama 3 B varyantı regresyonu

Önce yerel proje yedeği alındı. 6 referans ve 20 örneklem görüntüsü gerçek V3 ve SAM checkpoint’leriyle çalıştırıldı. Her iki koşuda ilk aşama sabit 3, alternatif kapalı. Kutu koordinatları 224×224 önizlemede, maske alanı aynı önizlemenin piksel sayısıdır.

17/26 görüntüde durum, kutu veya maske alanı değişti. 26 sınıf tahmini eşleşti. Eski hedef 14 kutu ve 14 maske üretti. Yeni hedef 14 kutu ve 13 maske üretti.

Değişen görüntüler Te-aug-me_7.jpg, Te-me_263.jpg, Te-pi_1.jpg, Te-gl_82.jpg, Te-gl_35.jpg, Te-gl_87.jpg, Te-gl_397.jpg, Te-me_106.jpg, Te-aug-me_15.jpg, Te-me_124.jpg, Te-me_270.jpg, Te-me_62.jpg, Te-pi_122.jpg, Te-pi_52.jpg, Te-pi_46.jpg, Te-pi_184.jpg, Te-pi_73.jpg.

| Görüntü | Gerçek sınıf | Swin tahmini | Eski durum | Eski kutu | Eski maske (px) | Yeni durum | Yeni kutu | Yeni maske (px) |
|---|---|---|---|---|---:|---|---|---:|
| Te-gl_22.jpg | glioma | glioma | kutu_yok | — | — | kutu_yok | — | — |
| Te-aug-me_7.jpg | meningioma | meningioma | hazır | [119, 73, 170, 140] | 1062 | kutu_yok | — | — |
| Te-gl_1.jpg | glioma | notumor | atlandı | — | — | atlandı | — | — |
| Te-me_263.jpg | meningioma | meningioma | hazır | [107, 72, 152, 119] | 748 | hazır | [53, 137, 85, 164] | 783 |
| Te-pi_1.jpg | pituitary | pituitary | kutu_yok | — | — | hazır | [16, 110, 71, 213] | 869 |
| Te-no_1.jpg | notumor | notumor | atlandı | — | — | atlandı | — | — |
| Te-gl_332.jpg | glioma | glioma | kutu_yok | — | — | kutu_yok | — | — |
| Te-gl_82.jpg | glioma | glioma | hazır | [76, 38, 136, 86] | 2670 | hazır | [106, 44, 131, 83] | 975 |
| Te-gl_35.jpg | glioma | glioma | hazır | [72, 22, 165, 80] | 4132 | hazır | [82, 29, 166, 83] | 3600 |
| Te-gl_87.jpg | glioma | meningioma | hazır | [105, 57, 131, 84] | 263 | hazır | [68, 72, 119, 138] | 2612 |
| Te-gl_397.jpg | glioma | glioma | hazır | [140, 44, 163, 69] | 523 | maske_yok | [172, 122, 192, 146] | — |
| Te-me_106.jpg | meningioma | meningioma | hazır | [44, 42, 118, 109] | 1876 | hazır | [145, 65, 179, 99] | 1000 |
| Te-aug-me_15.jpg | meningioma | meningioma | hazır | [118, 65, 164, 102] | 1334 | hazır | [165, 112, 216, 206] | 2710 |
| Te-me_124.jpg | meningioma | meningioma | hazır | [73, 56, 115, 101] | 926 | hazır | [75, 55, 115, 99] | 850 |
| Te-me_270.jpg | meningioma | meningioma | hazır | [101, 37, 138, 73] | 630 | hazır | [65, 156, 104, 218] | 382 |
| Te-me_62.jpg | meningioma | meningioma | hazır | [65, 76, 102, 115] | 331 | hazır | [160, 184, 182, 206] | 270 |
| Te-no_304.jpg | notumor | notumor | atlandı | — | — | atlandı | — | — |
| Te-no_267.jpg | notumor | notumor | atlandı | — | — | atlandı | — | — |
| Te-no_125.jpg | notumor | notumor | atlandı | — | — | atlandı | — | — |
| Te-no_226.jpg | notumor | notumor | atlandı | — | — | atlandı | — | — |
| Te-no_380.jpg | notumor | notumor | atlandı | — | — | atlandı | — | — |
| Te-pi_122.jpg | pituitary | pituitary | kutu_yok | — | — | hazır | [90, 0, 135, 40] | 230 |
| Te-pi_52.jpg | pituitary | pituitary | hazır | [118, 78, 178, 138] | 2928 | hazır | [118, 168, 207, 223] | 4452 |
| Te-pi_46.jpg | pituitary | pituitary | hazır | [21, 75, 122, 174] | 7550 | kutu_yok | — | — |
| Te-pi_184.jpg | pituitary | pituitary | hazır | [27, 27, 128, 114] | 6715 | kutu_yok | — | — |
| Te-pi_73.jpg | pituitary | pituitary | kutu_yok | — | — | hazır | [30, 34, 80, 83] | 1051 |

Altı referans görüntüde yedekteki fonksiyon ile yeni fonksiyonun Aşama 4 ve CNN float haritaları karşılaştırıldı. On iki SHA-256 çifti birebir eşleşti. Hash’ler [regresyon CSV’sinde](stage3_b_regression.csv). Model dosyalarının SHA-256 değerleri yedekle eşleşti. Segmentasyon fonksiyonunun tamamı yedekle satır satır eşleşti.

Regresyon komutu `.venv/Scripts/python.exe -B -u .venv/backups/stage3-b-20261120/regression.py`.

GitHub’a push yapılmadı. Kutu ve maskelerin anatomik doğruluğu doğrulanmadı.

## Uygulama ayarı ve doğrulama

Varsayılan `NEURO_SWIN_STAGE3_TARGET=block_norm1`, yani `layers[2].blocks[-1].norm1`. Eski hedef için çalışan sunucuyu durdurup aynı PowerShell oturumunda aşağıdaki komutları çalıştırın. Varsayılana dönmek için değişkeni kaldırıp sunucuyu yeniden başlatın.

```powershell
$env:NEURO_SWIN_STAGE3_TARGET = 'stage_output'
.\start.bat --port 8011
```

`/health`, yapılandırma değerini `swin_stage3_target`, hedef katmanı `swin_stage3_layer` alanında bildirir. Grad-CAM paneli aktif katmanı ve seçime uygun açıklamayı gösterir. PDF, sunucunun aktif hedefini Türkçe ve İngilizce rapora ekler. Bu metinler gerçek MR ile [Türkçe](stage3_b_tr.png) ve [İngilizce](stage3_b_en.png) tarayıcı koşusunda, PDF alanı da iki dilde metin çıkarılarak doğrulandı.

[README](../../README.md) Türkçe ve İngilizce olarak varsayılan katmanın notebook ile aynı olduğunu, canlı haritada çıkarım sırasında yumuşatma uygulandığını ve söz konusu notebook deney grafiklerinin yumuşatmasız olduğunu açıklar. Eski hedef seçildiğinde arayüz aşama çıkışına uygun açıklamayı gösterir.

| Kontrol | Sonuç |
|---|---|
| `.venv/Scripts/python.exe -B -m unittest discover -s tests -p 'test*.py'` | 4 test geçti |
| `.venv/Scripts/python.exe -B tests/smoke_web.py "../Brain_Tumor_MRI_Dataset.zip" --root http://127.0.0.1:8013` | CUDA ile tahmin, Grad-CAM, otomatik ve manuel SAM, atlanan otomatik segmentasyon, yükleme kontrolleri ve PDF üretimi geçti |
| Geri dönüş ayarı | Başlatma yapılandırması ve sağlık yanıtı işlevi iki değer için kontrol edildi. Eski hedefle ayrı tarayıcı koşusu doğrulanmadı. |
| Alternatif aşama | Mevcut açık kullanıcı seçimi korundu. 26 görüntülük regresyonda alternatif kapalıydı. |
| Aşama 4 ve CNN | Altı referans MR, 12 harita hash çifti birebir eşleşti |
| Kutu ve maske anatomik doğruluğu | Doğrulanmadı |


### Canlı 8011 kontrolü

İlk istekte 8011 portu bağlantıyı reddetti. Sunucu bu portta başlatıldı ve [canlı sağlık yanıtı](http://127.0.0.1:8011/health) yeniden okundu. Sonuç `status=hazır`, `device=cuda`, `swin_profile=v3`, `swin_stage3_target=block_norm1`, `swin_stage3_layer=layers[2].blocks[-1].norm1`. Aynı sunucunun ana sayfasında Grad-CAM katman alanı bulundu. Yukarıdaki ekran görüntüleri ve smoke testi 8013 üzerindeki önceki doğrulama koşusundan.
