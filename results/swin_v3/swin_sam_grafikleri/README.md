# Swin-Tiny V3 ve SAM grafikleri

## Sunumda öne çıkarılacaklar

| Grafik | İçerik |
| --- | --- |
| [01 · V3 deney karşılaştırması](01_swin_v3_deney_karsilastirmasi.png) | Başlangıç, yerel RTX 3050 ve Colab T4 accuracy, glioma recall ve glioma → tümör yok hatası. |
| [02 · Glioma hataları](02_glioma_hatalari.png) | 383 glioma görüntüsünün doğru, tümör yok ve diğer yanlış tahminlere ayrılması. |
| [10 · Dört sınıfta Grad-CAM ve SAM](10_sam_ve_gradcam_dort_sinif.png) | Özgün MR, Stage 3, Stage 4 ve otomatik SAM maskesi. |
| [11 · Manuel ve otomatik SAM](11_sam_otomatik_manuel_kutu.png) | Aynı meningioma görüntüsünde iki kutu yöntemi. |

## Ayrıntılı Swin grafikleri

- [03 · Sınıf bazında precision, recall ve F1](03_sinif_bazinda_precision_recall_f1.png)
- [04 · Karmaşıklık matrisleri ve hücre farkları](04_karisiklik_matrisi_ve_fark.png)
- [05 · Epoch bazında accuracy ve glioma recall](05_epoch_accuracy_ve_glioma_recall.png)
- [06 · ROC ve AUC](06_roc_auc.png)
- [07 · Precision–recall ve ortalama precision](07_precision_recall.png)
- [08 · Doğru ve yanlış tahminlerin softmax dağılımı](08_dogru_yanlis_softmax_dagilimi.png)
- [09 · Softmax güvenilirlik grafiği](09_softmax_kalibrasyonu.png)
- [12 · Otomatik SAM maske alanı ve kalite tahmini](12_sam_maske_alani_ve_kalite_skoru.png)

## Kaynaklar ve kapsam

01 ve 02 numaralı grafiklerin Colab T4 değerleri, Colab ekranında doğrulanan %93,98 accuracy, %81,72 glioma recall ve 39 glioma → notumor hatasına dayanır. Glioma destek sayısı 383. 313 doğru ve 31 diğer yanlış, bu sayılardan hesaplandı. Colab T4’ün tam karmaşıklık matrisi ya da tek tek olasılıkları yerelde bulunmadığından, 03, 04 ve 06–09 numaralı grafikler başlangıç V3 ve yerel RTX 3050 checkpoint’lerinin gerçek 1.579 görüntülük V3 çıkarımlarını kullanır. 05 numaralı grafik yerel RTX 3050 eğitim kaydını kullanır.

10–12 numaralı grafikler başlangıç Swin V3 ve mevcut SAM ViT-B checkpoint’iyle yerelde üretilen dört örneğe dayanır. SAM skoru modelin kalite tahminidir. Uyumlu referans maskeler bulunmadığı için Dice ve IoU hesaplanmadı. Görüntü işleme yöntemi, Swin `notumor` dediğinde de parlak bir aday bulabilir. Bu durum ilgili örneklerde işaretlendi. V3 test kümesinin hasta düzeyinde bağımsızlığı doğrulanmadı.

[Önceki veri artırma, ağırlık ve mimari grafikleri](../colab_t4_ek_analiz/) ve [önceki eğitim kaybı eğrileri](../finetuning/local_run/training_curves.png) mevcut klasörlerinde.
