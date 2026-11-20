# Swin-Tiny V3 · yerel fine-tuning ve SAM doğrulaması

**18 Kasım 2026 · RTX 3050 Laptop GPU · PyTorch 2.6.0+cu124 · timm 1.0.22**

## Eğitim ve test

Yerel V3 checkpoint’i, GitHub’daki V3 notebook’unda belgelenen sınıf sırası ile yüklendi. Ham state dict mimari ve tensör boyutları açısından strict yüklemeyi geçti. Aynı V3 test listesinin yeniden çalıştırılması, paylaşılan başlangıç confusion matrix’indeki 16 hücrenin tamamını yeniden üretti. Mevcut V3 ve eski Swin checkpoint’leri korundu.

| Ayar | Yerel çalışma |
| --- | --- |
| Mimari | `swin_tiny_patch4_window7_224`, dört sınıf |
| Başlangıç | `models/swin_tiny_v3_best.pth` |
| Sınıf sırası | glioma, meningioma, notumor, pituitary |
| Ön işleme | RGB, 224 × 224, ImageNet ortalama ve standart sapması. Eğitimde ±10° döndürme |
| Veri | Eğitim 3.870, doğrulama 994, test 1.579 |
| Seed | 42, Python / NumPy / PyTorch |
| Optimizasyon | AdamW, LR 5e-6, weight decay 0.01, sınıf ağırlıkları [1.5, 1, 1, 1] |
| GPU ayarı | FP16 autocast ve GradScaler. Mikro batch 8, gradient accumulation ile etkin batch 16. Gözlenen en yüksek ayrılmış PyTorch CUDA belleği 1.242 MiB |
| Seçim | Validation accuracy. Eşitlikte glioma recall ve daha az glioma → notumor hatası |
| Early stopping | En fazla 5 epoch, 2 epoch sabır. 3 epoch çalıştı, 1. epoch seçildi |
| Seçilen doğrulama | Accuracy %98,39, glioma recall %97,86, glioma → notumor 0 |
| Yeni checkpoint | `models/swin_tiny_v3_finetuned_best.pth` |

Test CSV’si eğitim ve checkpoint seçimi tamamlandığında açıldı. Eğitim kaybı ağırlıklı, doğrulama kaybı ağırlıksız çapraz entropiyle raporlandı. CSV özetleri, epoch ölçümleri ve test metrikleri aşağıdaki JSON dosyasında kayıtlı.

| Deney | Test accuracy | Glioma recall | Doğru glioma | Glioma → notumor |
| --- | ---: | ---: | ---: | ---: |
| V3, yerelde yeniden doğrulandı | %93,86 | %82,77 | 317/383 | 49 |
| Paylaşılan önceki fine-tuning | %94,11 | %81,98 | 314/383 | 39 |
| Bu yerel fine-tuning | %94,05 | %81,98 | 314/383 | 41 |

Yerel model, başlangıca göre 3 ek görüntüyü doğru sınıflandırdı ve glioma → notumor hatasını 8 azalttı. Toplam doğru glioma tahmini 3 düştü. Yerel macro F1 **%93,92**, weighted F1 **%93,96**. `Te-gl_1.jpg` için başlangıç ve yerel fine-tuned model sırasıyla %99,847 ve %99,840 softmax ile notumor dedi. Bu görüntü eğitim listesine eklenmedi. V3 test listesi hasta düzeyinde bağımsız doğrulanmadı.

| Yerel test sınıfı | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| glioma | %100,00 | %81,98 | %90,10 |
| meningioma | %93,70 | %93,94 | %93,82 |
| notumor | %87,34 | %100,00 | %93,24 |
| pituitary | %97,32 | %99,75 | %98,52 |

Yerel confusion matrix, satırlar gerçek etiketi ve sütunlar tahmini gösterecek şekilde:

```text
[[314, 24, 41,  4],
 [  0,372, 17,  7],
 [  0,  0,400,  0],
 [  0,  1,  0,399]]
```

- [Eğitim eğrileri](../../results/swin_v3/finetuning/local_run/training_curves.png) · [Karşılaştırmalı matrisler](../../results/swin_v3/finetuning/local_run/confusion_comparison.png)
- [Deney kaydı](../../results/swin_v3/finetuning/local_run/metrics.json) · [Metrik karşılaştırması](../../results/swin_v3/finetuning/local_run/comparison.csv) · [Sınıflandırma raporu](../../results/swin_v3/finetuning/local_run/classification_report.csv)
- [Checkpoint doğrulaması](../../results/swin_v3/finetuning/local_run/checkpoint_verification.json) · [SAM ve Grad-CAM denetimi](../../results/swin_v3/finetuning/local_run/segmentation_checks.json)

## Web ve segmentasyon doğrulaması

- Eski profil için PNG/JPG tahmin, Stage 3 Grad-CAM otomatik kutusu, manuel SAM, notumor atlama ve hata kontrolleri gerçek HTTP üzerinden geçti. Ek HTTP testinde Stage 3 ve Stage 4 farklı ısı haritaları üretti. Aynı görüntü işleme kutusu her iki aşamaya da gönderildi. Boş görüntüde aday bulunamadı ve maske dönmedi.
- V3 profili ile dört sınıftan birer görüntüde iki Grad-CAM aşaması ve görüntü işleme + SAM çalıştırıldı. Meningioma ayrıca manuel kutuyla test edildi. **Notumor örneğinde yöntem bir bölge seçti.** Bu ve glioma örneğindeki notumor sınıflandırması için uyarı döndü. Buna göre otomatik kutu, deneysel aday üretimi olarak yorumlanmalı. Önceki paylaşılan iki SAM kutusu birebir yeniden üretilmedi. Yerel kutular ayrı JSON kaydında.
- Python derleme ve JavaScript sözdizimi kontrolleri geçti. Yeni tarayıcı seçicileri görsel olarak tarayıcıda tekrar test edilmedi. Referans maskeler eşleştirilmediği için Dice/IoU hesaplanmadı.

## Aktarım

Drive bağlantısı oturum açma sayfasına yönlendi. Bu oturumda Drive klasörü okunamadı veya değiştirilemedi. İstenirse yerel yeni checkpoint Drive’daki `modeller/` altında benzersiz bir adla ve rapor çıktıları `results/swin_v3/finetuning/local_run/` altında saklanabilir. Paylaşılan önceki fine-tuned modelin Drive’da bulunup bulunmadığı doğrulanmadı.

GitHub `main` ağacı salt okunur incelendi. Oradaki V3 notebook’u ve mevcut sonuçlar korundu. Çalışma klasöründe Git metadata’sı bulunmadığından yerel değişiklikler commit edilmedi. Push yapılmadı. Paylaşım için önerilen kapsam eğitim/denetim betikleri, backend ve arayüz kodu, dokümantasyon ve toplu metrik CSV/PNG dosyalarıdır. Model ağırlıkları ile ham MR ve ZIP dosyaları yerelde tutulmalıdır.
