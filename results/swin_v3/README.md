# Swin-Tiny V3 sonuçları

Bu klasörde 1.579 görüntülük V3 test listesiyle elde edilen sonuçlar bulunur. İlk, 1.600 görüntülük Swin deneyi için [ana deney tablosuna](../../README.md#swin-deneyleri) bakın.

| Deney | Dosyalar |
| --- | --- |
| V3 başlangıç | [Özet CSV](swin_v3_summary.csv) · [Sınıflandırma raporu](classification_report.txt) · [Karmaşıklık matrisi](confusion_matrix.png) · [Test tahminleri](test_predictions.csv) |
| V3 başlangıç eğitimi | [Eğitim geçmişi](training_history.csv) · [Eğitim eğrileri](training_curves.png) |
| Yerel fine-tuning | [Karşılaştırma CSV](finetuning/local_run/comparison.csv) · [Sınıf bazında metrikler](finetuning/local_run/classification_report.csv) · [Karşılaştırmalı matrisler](finetuning/local_run/confusion_comparison.png) |
| Yerel fine-tuning eğitimi | [Deney ayarları ve metrikler](finetuning/local_run/metrics.json) · [Eğitim geçmişi](finetuning/local_run/training_history.csv) · [Eğitim eğrileri](finetuning/local_run/training_curves.png) |

Colab T4 çalışmasının ayrıntılı çıktıları Drive’da `results/swin_v3/finetuning/colab_20261118T160853420387Z/` altında. Özet metrikler [ana deney tablosunda](../../README.md#swin-deneyleri).

Aynı adlı eğitim grafikleri farklı deneylere aittir. Önceki paylaşılan fine-tuning değerleri ve yerel çalışma arasındaki farklar [Swin V3 raporunda](../../SWIN_V3_TRAINING_REPORT.md) açıklanıyor.
