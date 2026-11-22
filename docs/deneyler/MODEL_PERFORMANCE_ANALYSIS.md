# NEURO AI LAB · Model performans analizi

**18 Kasım 2026 · 1.600 test görüntüsü · Her sınıftan 400 görüntü**

Swin-Tiny, aynı test kümesinde **%93,25 accuracy** ve **%93,19 macro F1** ile önde. Glioma için en yüksek recall da Swin-Tiny’de (%80,75). Buna rağmen, en az %99 softmax skoru verdiği 40 yanlışın 34’ü glioma. Veri denetiminde 114 test görüntüsünün eğitim veya doğrulama tarafında piksel düzeyinde aynısı bulundu. Bu nedenle yeniden eğitim kararından önce veri ve etiket denetimi öncelikli.

## Değerlendirme ve ön işleme

`Brain_Tumor_MRI_Dataset.zip` arşivindeki `Testing/` klasörünün tamamı değerlendirildi. Gerçek etiket, test alt klasöründen alındı. Sınıf sırası **0 glioma, 1 meningioma, 2 notumor, 3 pituitary**. `cnn_best.pth` ve `swin_tiny_best.pth` mevcut mimarilere `strict=True` ile yüklendi. Swin’in sınıf sırası checkpoint meta verisiyle de doğrulandı. RTX 3050 Laptop GPU, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124 ve timm 1.0.22 kullanıldı. Modeller `eval()` modunda, 16 görüntülük gruplarla sırayla çalıştı. Tahmin, dört logite uygulanan softmax’ın en büyük bileşeni olarak alındı.

| Notebook | Eğitim dönüşümleri | Test dönüşümleri |
| --- | --- | --- |
| CNN, hücre 9 | Gri tonlama (`L`), 224×224, `ToTensor`, ortalama ve standart sapma 0,5 | Aynı dönüşümler |
| Swin, hücreler 10 ve 12 | RGB, 224×224, ±10° döndürme, `ToTensor`, ImageNet normalizasyonu | RGB, 224×224, `ToTensor`, aynı normalizasyonu kullanan `val_transform` |

Swin için ImageNet ortalaması `[0.485, 0.456, 0.406]`, standart sapması `[0.229, 0.224, 0.225]` kullanıldı. [CNN](../../notebooks/CNN.ipynb) · [Swin](../../notebooks/Swin_Transformer_SAM.ipynb)

## Genel sonuçlar

| Model | Doğru / 1.600 | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN | 1.228 | %76,75 | %79,85 | %76,75 | %75,88 | %75,88 |
| Swin-Tiny | 1.492 | **%93,25** | **%93,88** | **%93,25** | **%93,19** | **%93,19** |

Sınıf destekleri eşit olduğu için macro recall accuracy’ye, macro F1 weighted F1’a eşit. Swin-Tiny, CNN’den 264 daha fazla görüntüyü doğru sınıflandırdı.


## Sınıf bazında precision, recall ve F1

| Sınıf | Model | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| glioma | CNN | %97,35 | %55,00 | %70,29 |
| glioma | Swin-Tiny | %97,88 | **%80,75** | **%88,49** |
| meningioma | CNN | %67,53 | %58,75 | %62,83 |
| meningioma | Swin-Tiny | **%84,52** | **%98,25** | **%90,87** |
| notumor | CNN | %66,33 | **%100,00** | %79,76 |
| notumor | Swin-Tiny | **%93,66** | %99,75 | **%96,61** |
| pituitary | CNN | %88,18 | %93,25 | %90,64 |
| pituitary | Swin-Tiny | **%99,47** | %94,25 | **%96,79** |

Swin-Tiny dört sınıfta da CNN’den daha yüksek F1 verdi. CNN’in tümör yok duyarlılığı %100, kesinliği %66,33.

## Karmaşıklık matrisleri ve glioma hataları

Satırlar gerçek test etiketini, sütunlar model tahminini gösterir. Her satırda 400 görüntü vardır.


| Gerçek etiket glioma (400) | Doğru glioma | meningioma | notumor | pituitary | Toplam yanlış |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN | 220 | 98 | 71 | 11 | 180 |
| Swin-Tiny | 323 | 53 | 24 | 0 | 77 |

- Glioma → tümör yok hata sayısı CNN’de 71 ve Swin-Tiny’de 24.
- Eğitim veya doğrulama ile pikseli aynı glioma görüntüleri çıkarıldığında kalan 395 glioma üzerinde recall sırası korundu. CNN 216/395 (%54,68), Swin-Tiny 318/395 (%80,51).

## Yanlış ama yüksek olasılıklı tahminler

Her hücre, ilgili eşikte **yanlış tahmin / eşiği geçen tüm tahmin** sayısını gösterir. Eşikler en yüksek softmax skoruna uygulanır ve iç içedir.

| Model | ≥ %90 | ≥ %95 | ≥ %99 | Yanlışlarda ortalama en yüksek skor |
| --- | ---: | ---: | ---: | ---: |
| CNN | 56 / 828 | 28 / 581 | 7 / 210 | %72,27 |
| Swin-Tiny | 67 / 1.525 | 52 / 1.497 | 40 / 1.436 | %88,64 |

Yüksek skorlu yanlışların glioma olanları CNN’de 32/56 ve Swin-Tiny’de 53/67. Yüksek model skoru doğruluğu güvenceye almıyor.

| Model | En yüksek skorlu yanlış | Test etiketi → tahmin | En yüksek skor |
| --- | --- | --- | ---: |
| CNN | Te-gl_43.jpg | glioma → notumor | %99,9493 |
| Swin-Tiny | Te-gl_68.jpg | glioma → meningioma | %99,9949 |


## Te-gl_1.jpg · Test etiketi glioma

| Model | glioma | meningioma | notumor | pituitary | Tahmin |
| --- | ---: | ---: | ---: | ---: | --- |
| CNN | %0,24091 | %36,84943 | **%40,30476** | %22,60491 | notumor |
| Swin-Tiny | %0,00789 | **%99,98814** | %0,00348 | %0,00049 | meningioma |

Bu örnekte CNN tümör yok, Swin-Tiny meningioma tahmini verdi. Tekil sonuç genel glioma performansını temsil etmez.

## Eğitim/doğrulama örtüşmesi ve bağımsızlık

Yerelde özgün bölünme CSV’leri bulunmadı. Swin notebook’undaki sınıf sırası ve sıralı dosya yollarından, `test_size=0.20`, `random_state=42` ve sınıfa göre stratification ile **4.480 eğitim / 1.120 doğrulama** yolu yeniden oluşturuldu. CNN notebook’u Drive’daki aynı adlı listeleri okuyor. Yeniden oluşturulan üyeliğin özgün listelerle tek tek aynı olduğu veya bu listelerin her checkpoint’in eğitiminde aynen kullanıldığı doğrulanamadı. Birleşik 5.600 görüntülük `Training/` ile `Testing/` karşılaştırması bu ayrıma bağlı değildir.

Arşivdeki dosyalar baz ad, özgün baytların SHA-256 özeti ve boyutla birlikte çözümlenmiş RGB piksellerinin SHA-256 özetiyle karşılaştırıldı. Pikseli birebir aynı olmayan çiftlerde, 64 bit dHash Hamming uzaklığı ≤4 ve 64×64 gri tonlamalı önizleme ortalama mutlak piksel farkı ≤5/255 koşullarını sağlayanlar benzerlik adayı olarak kaydedildi.

| Karşılaştırma | Aynı baz ad, çift | Aynı bayt, çift | Aynı RGB pikseli, çift | Olası benzerlik, çift |
| --- | ---: | ---: | ---: | ---: |
| Test ↔ eğitim | 0 | 0 | 94 | 549 |
| Test ↔ doğrulama | 0 | 0 | 27 | 180 |
| Doğrulama ↔ eğitim | 0 | 78 | 88 | 436 |
| Test kümesi kendi içinde | 0 | 17 | 22 | 166 |

- Eğitim/doğrulama birleşimine karşı **114 benzersiz test görüntüsünün (%7,13)** çözümlenmiş pikselleri birebir aynı. Bunların 100’ü meningioma, 9’u notumor ve 5’i glioma. Test kümesinin kendi içinde 22 piksel-eş çiftine toplam 41 dosya katılıyor.
- Kesin eşler çıkarıldıktan sonra **330 ek test görüntüsünde** benzerlik adayı var. Bunların 308’i notumor, 19’u glioma ve 3’ü meningioma. Aday sayıları algoritmanın eşiğine bağlı.
- Güvenilir hasta veya çalışma kimliği sağlanmadığı için **hasta düzeyinde bağımsızlık doğrulanamadı**. İçerik eşleme farklı kesitleri, kırpmaları ve diğer dönüşümleri kaçırabilir.

### Örtüşme hariç tutulduğunda

| Alt küme | Görüntü | CNN accuracy / macro F1 | Swin-Tiny accuracy / macro F1 |
| --- | ---: | ---: | ---: |
| Tam test | 1.600 | %76,75 / %75,88 | %93,25 / %93,19 |
| Eğitim/doğrulama ile piksel-eş olanları çıkar | 1.486 | %81,36 / %79,99 | %93,00 / %92,73 |
| Ek olarak benzerlik adaylarını çıkar | 1.156 | %76,56 / %72,99 | %91,18 / %90,16 |

Model sıralaması her iki alt kümede de korundu. Eleme sınıf dağılımını değiştirdi. İkinci elemede 400 notumor görüntüsünden 83’ü kaldı. Bu değerler duyarlılık analizi olarak okunmalı. Herhangi bir alt kümenin hasta düzeyinde bağımsız olduğunu göstermiyorlar.

## Yeniden eğitim kararı

Swin-Tiny, tam testte ve örtüşme elemelerinden sonra CNN’den daha yüksek genel metrikler verdi. CNN’de %55 glioma duyarlılığı ve 71 glioma → tümör yok hatası, ileri inceleme için önceliklidir.

Model değiştirmeden önce, benzer ve çelişkili örnekleri kaynak veride inceleyip hasta düzeyinde ayrılmış bir test seti kurmak daha güçlü bir karar zemini sağlar. Bu bağımsız değerlendirmede glioma recall hedefin altında kalırsa yeniden eğitim gerekçesi oluşur. Sorun ağırlıkla yüksek softmax skorlarında kalırsa, bağımsız doğrulama kümesinde skor kalibrasyonunu değerlendirmek ayrı bir adımdır.

## Çıktılar

- [Tüm tahminler ve dört sınıf olasılıkları](../../performance_analysis/predictions.csv) · 3.200 satır
- [En az %90 skorlu yanlış tahminler](../../performance_analysis/high_confidence_errors.csv)
- [Örtüşme çiftleri, benzerlik ölçüleri ve etiket uyuşmazlıkları](../../performance_analysis/overlap_pairs.csv)
