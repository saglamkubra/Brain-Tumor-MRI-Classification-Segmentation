# Grad-CAM ve SAM · Koordinat ve aşama denetimi

## Bulgular ve düzeltmeler

- Git geçmişindeki [46847fa](https://github.com/saglamkubra/Brain-Tumor-MRI-Classification-Segmentation/tree/46847fad8dcd0d97b4f09fa7178c96ea5fce99ce) ile güncel akış aynı görüntülerde ve aynı checkpoint'lerle çalıştırıldı. Önceki yerel uygulama kodu ayrıca güvenli çalışma kopyasına alındı. Sınıflandırma ön işlemesinde `Models.tensor` kullanıldı ve değiştirilmedi. Güncel ve eski Grad-CAM sınıfları altı görüntünün hepsinde normal Swin tahminiyle eşleşti.
- Eski kod görseli 224 × 224 kareye esnetiyor ve SAM'ı bu yeniden örneklenmiş kare üzerinde çalıştırıyordu. SAM dönüşümünün iki kez uygulandığına dair kanıt yok. Güncel kod orijinal RGB'yi SAM'e veriyor. 224 × 224 önizlemede görüntünün oranı korunuyor. Grad-CAM ızgarası da bu önizlemedeki aynı dikdörtgene geri taşınıyor. Kutu ve noktalar SAM'e verilmeden önce orijinal görüntü koordinatlarına dönüştürülüyor. SAM'in kendi kodlayıcı dönüşümü bir kez uygulanıyor ve ürettiği maske ortak önizlemeye en yakın komşu ile aktarılıyor.
- Eski kutu, sabit elips içindeki 90. yüzdelik eşik üstü bileşenleri birleştiriyordu. Uzak odaklar kutuyu büyütebiliyordu. Güncel yöntemde ön plan yaklaşık bir görüntü alanıdır. Eşikleme, kapatma, en büyük dış konturun doldurulması ve erozyon kullanılır. Grad-CAM bileşenleri için görüntü kenarı ayrıca erozyonla elenir. Dağınık, benzer güçte birden fazla odak olduğunda veya kutu fazla geniş/dar olduğunda yöntem çekimser kalır. Yüzdelik ve göreli eşikler önceki yerel yönteme dayanır. Bu ekran görüntülerine göre sayısal eşik taraması yapılmadı.
- SAM alternatifi hem API hem arayüz varsayılanında kapalıdır. Kullanıcı açarsa karşı aşama ile ortak odak denenir. Uygun ortak odak çıkmazsa karşı aşamanın kutusu değerlendirilebilir. Kartta gösterilen Grad-CAM aşaması kullanıcı seçimine bağlıdır. SAM kartı seçilen aşamanın kutu üretememe nedenini ve varsa gerçek kutu kaynağını gösterir. Manuel SAM ayrı çalışır.
- Dört panel aynı ölçekli kare önizlemeler kullanır. Yan yana ana düzende hiçbir MR kırpılmaz ya da gerilmez. İkili Grad-CAM karşılaştırması aynı boyutu korumak için Grad-CAM kartında alt alta görünür. Orijinal ile işlenmiş çıktılar büyük karşılaştırma penceresinde açılabilir.

## Grad-CAM matematiği

Aşama 3, `model.layers[2]` çıkışını (1 × 14 × 14 × 384), Aşama 4, `model.layers[3].blocks[-1].norm1` çıkışını (1 × 7 × 7 × 768) kullanır. Aktivasyon ve gradyan NHWC düzeninden NCHW düzenine çevrilir. Model değerlendirme modundadır. Hedef, aynı görüntünün Swin sınıf logitidir. Grad-CAM kanal başına uzamsal ortalama gradyanı ağırlık kabul eder, aktivasyonları ağırlıklı toplar ve pozitif katkıyı alır. Altı çevirme ve yoğunluk varyantı ters hizalanıp ortalanır. Bu çalışmada hedef katmanlar ve gradyan formülü önceki yerel uygulamayla aynı kaldı. Hesaptaki yeni işlem, haritayı orijinal en-boy oranına göre gösterim koordinatlarına taşımaktır.

Te-gl_22 için her iki aşamada 12 gerçek aktivasyon/gradyan çifti gözlemlendi ve iki çalıştırma sonunda hook sayısı eski değerine döndü. Haritalar 224 × 224, sonlu ve tekrarda tutarlıydı. CPU ve CUDA'da Aşama 3 sınıfı eşleşti. En büyük mutlak harita farkı 0,00000188, ortalama fark 0,00000035 oldu. Bu karşılaştırma tek görüntünün Aşama 3'ü içindir.

Açık renkli alt ve yan dokulardaki aktivasyonlar, gösterim öncesi Grad-CAM değerlerini yansıtır. Te-pi_1 örneğinde iki katman farklı dokulara yayılıyor. Kütüphane ve hizalama kontrollerinde bu durumu açıklayacak kanal, hook veya koordinat hatası bulunmadı. Bu kontrol modelin çene veya boyuna nedensel olarak dayandığını kanıtlamaz. Ön plan maskesi anatomik beyin segmentasyonu içermediğinden bütün kafa dışı dokuları güvenilir şekilde ayırmaz.

## Gerçek model çıktıları

| Örnek | Eski otomatik | Güncel varsayılan | Alternatif açıkken |
|---|---:|---:|---|
| Te-gl_22.jpg | 21.723 piksel | Kutu reddedildi, fazla geniş | Ortak odak, 1.467 piksel |
| Te-aug-me_7.jpg | 16.967 piksel | Aşama 3, 1.062 piksel | Aşama 3 |
| Te-gl_1.jpg | Atlandı | Atlandı | Atlandı |
| Te-me_263.jpg | 17.781 piksel | Aşama 3, 748 piksel | Aşama 3 |
| Te-pi_1.jpg | 19.767 piksel | Kutu çıkmadı, odak ayrışmadı | Aşama 4, 3.774 piksel |
| Te-no_1.jpg | Atlandı | Atlandı | Atlandı |

Piksel sayıları 224 × 224 gösterim alanına aittir. Hipofiz örneği olarak arşivdeki Te-pi_1 kullanıldı. Uygulama yüklenen görüntü adını kaydetmediği ve bu istekte ekran görüntüsü bulunmadığı için bunun son ekran görüntüsünün dosyası olduğu doğrulanamadı. Te-pi_1'de alternatif Aşama 4 aday maskesi yan tarafa kayıyor. Te-gl_1'de iki model de veri setindeki glioma etiketiyle uyuşmayan tümör yok sonucunu verdi. Bu nedenle otomatik SAM atlandı. Manuel kutuyla aynı dosyada maske üretildi.

## Testler ve yeniden çalıştırma

- Dört birim testi ve mevcut gerçek model HTTP duman testi geçti.
- Chrome'da 1440 ve 390 piksel görünüm, aynı panel ölçeği, Aşama 3 varsayılanı, tek/çift aşama, alternatifin açık ve kapalı akışı, otomatik kutunun manuel alana aktarılması, klavyeyle kutu düzenleme, büyük görüntüleyici ve opaklık doğrulandı.
- Gerçek dikey Te-gl_1 (354 × 442) ve yatay Te-gl_103 (541 × 477) MR'da önizleme ve iki Grad-CAM katmanının hizalanması doğrulandı. Te-gl_1 manuel SAM maskesinin boşluklara taşmadığı, kutunun orijinal koordinatlara doğru dönüştüğü ve maske ile kutu dışında RGB piksellerinin değişmediği doğrulandı.
- Checkpoint'ler değişmedi. Referans segmentasyon maskesi bulunmadığından tümör sınırı doğruluğu, Dice ve IoU ölçülmedi.

Çalıştırmak için PowerShell'de `.start.bat --port 8011`. [Görseller](gradcam_sam_once_sonra.png) ve [ham ölçümler](measurements.json) bu klasörde. Karşılaştırmayı yeniden üretmek için `experiments/swin_v3/audit_gradcam_sam.py` arşiv, Git geçmişi ve çıktı dizini parametreleriyle çalıştırılabilir.
