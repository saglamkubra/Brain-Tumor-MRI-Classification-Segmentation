# NEURO AI LAB checkpoint test raporu

18 Kasım 2026 · Windows · Python 3.12.0

Dört checkpoint de notebook mimarileriyle `strict=True` kullanılarak eksik veya fazla parametre anahtarı olmadan yüklendi. Test girdisi, üst klasördeki `Brain_Tumor_MRI_Dataset.zip` içinden okunan 512×512 boyutlu `Testing/meningioma/Te-me_263.jpg` dosyasıydı.

| Model | Checkpoint | Bayt | MiB | Kayıt epoch'u |
| --- | --- | ---: | ---: | ---: |
| CNN | `cnn_best.pth` | 1.346.327 | 1,28 | 18 |
| ViT-B/16 | `vit_best.pth` | 343.267.949 | 327,37 | 10 |
| Swin-Tiny | `swin_tiny_best.pth` | 110.158.666 | 105,06 | 3 |
| SAM ViT-B | `sam_vit_b_01ec64.pth` | 375.042.383 | 357,67 | — |

## CPU çıkarımı

PyTorch 2.6.0+cpu, torchvision 0.21.0+cpu, timm 1.0.22 ve segment-anything 1.0 kullanıldı. CNN sınıfı doğrudan notebook tanımından alındı. ViT ve Swin için notebook'taki dört çıkışlı mimariler oluşturuldu. Spesifik checkpoint'lerin tüm ağırlıkları yüklendi.

Görüntü CNN için gri tonlamaya, ViT ve Swin için RGB'ye dönüştürüldü. Üçü de 224×224 boyutlandırma ve notebook'larındaki değerlendirme normalizasyonunu kullandı. Her çıktı dört sonlu logit üretti ve softmax toplamı 1 olarak doğrulandı. Swin checkpoint'inin sınıf listesi notebook'lardaki sırayla eşleşti.

| Model | 0 glioma | 1 meningioma | 2 notumor | 3 pituitary |
| --- | ---: | ---: | ---: | ---: |
| CNN | %6,27230 | **%84,45224** | %3,58914 | %5,68633 |
| ViT | %0,66418 | **%98,98257** | %0,00428 | %0,34898 |
| Swin | %0,00041 | **%99,99818** | %0,00137 | %0,00004 |

SAM, 224×224 RGB görüntü üzerinde notebook'taki `[115, 85, 160, 135]` manuel kutusuyla üç adet 224×224 maske üretti. Kalite skorları 0,861711, 0,909220 ve 0,935443 idi. En yüksek skorlu maske 732 piksel içeriyordu.

Bu tek görüntüyle yapılan bir yükleme ve çıkarım testidir. Genel doğruluk, Grad-CAM, otomatik SAM kutusu ve segmentasyon doğruluğu bu aşamanın kapsamı dışındaydı.

## CUDA çıkarımı

NVIDIA GeForce RTX 3050 Laptop GPU (4 GiB), NVIDIA sürücüsü 566.07, PyTorch 2.6.0+cu124 ve torchvision 0.21.0+cu124 kullanıldı. `torch.cuda.is_available()` True döndürdü ve GPU üzerinde tensör işlemi tamamlandı. Dört modelin çıkarımı `cuda` cihazında ayrı ayrı başarıyla çalıştı.

| Model | 0 glioma | 1 meningioma | 2 notumor | 3 pituitary |
| --- | ---: | ---: | ---: | ---: |
| CNN | %6,27087 | **%84,45063** | %3,58951 | %5,68900 |
| ViT | %0,66421 | **%98,98254** | %0,00428 | %0,34898 |
| Swin | %0,00041 | **%99,99818** | %0,00137 | %0,00004 |

SAM aynı kutuyla üç maske üretti. Kalite skorları 0,861736, 0,909229 ve 0,935452 idi. En yüksek skorlu maske 732 piksel içeriyordu.

## GitHub

[`.gitignore`](../../.gitignore) dosyasına `/models/` eklendi. `git check-ignore` ile kuralın dört dosyaya da uygulandığı doğrulandı. GitHub deposunun yayımlanan `main` dalındaki 23 commit tarandı. Uç commit `4bb080a6f8930c848feeb9588c37b8ae53b16e7d` idi. Geçmişte checkpoint yolları aranıp dört dosyanın Git blob kimliği geçmişteki nesnelerle karşılaştırıldı. Her iki kontrolde de eşleşme çıkmadı. [GitHub deposu](https://github.com/saglamkubra/Brain-Tumor-MRI-Classification-Segmentation)
