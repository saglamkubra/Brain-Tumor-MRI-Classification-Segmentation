# Grad-CAM ve SAM doğrulama raporu

## Karşılaştırma

Önceki kayıtlı uygulama [Git revizyonu 46847fa](https://github.com/saglamkubra/Brain-Tumor-MRI-Classification-Segmentation/tree/46847fad8dcd0d97b4f09fa7178c96ea5fce99ce) üzerinden alındı. Çalışmaya başlamadan önce yerel uygulama dosyalarının bir kopyası `.venv/gradcam-sam-before-audit/` altına kaydedildi. Git revizyonundaki kutu ve Grad-CAM kodu ile bu karşılaştırmanın güncel kodu aynı süreçte, aynı CUDA cihazında, aynı Swin-Tiny V3 ve SAM ağırlıklarıyla çalıştırıldı. İki yönteme aynı RGB MR verildi. Sınıflandırma ön işlemesi için aynı `Models.tensor` ve 224 × 224 ImageNet normalizasyonu kullanıldı. Her iki yöntemin SAM önizlemesi piksel düzeyinde eşleşti.

Eski kod, sabit eliptik pencerenin içindeki en üst %10 aktivasyonu alıp yeterli büyüklükteki bütün bağlı bileşenleri tek kutuya birleştiriyordu. Birbirinden uzak odaklar kutuyu büyütebiliyordu. SAM, kutudan üretilen maskeler içinde en yüksek model kalite skorunu seçiyordu. Güncel yöntem görüntüden çıkarılan ön planı kullanır, yumuşatılan Grad-CAM üzerinde göreli eşik ve 90. yüzdelik uygular, benzer güçte birden çok odak varsa çekimser kalır, tek bileşen seçer ve kutu boyutunu sınırlar. Ön plan, eşikleme, morfolojik kapatma, en büyük dış konturu doldurma ve erozyon ile elde edilen görüntü alanıdır. Anatomik bir beyin segmentasyonu içermez. SAM aday maskeleri kutuya ve işaretlere uyum ve kapsanan alan açısından elenir. Koordinatlar x1, y1, x2, y2 sırasıyla 224 × 224 önizlemede tutulur. SAM’in kendi `ResizeLongestSide` dönüşümü kutuyu ve noktaları kodlayıcı koordinatlarına taşır, çıktıyı 224 × 224’e geri eşler. Tarayıcı koordinatları bu önizlemeye ölçekler.

## Grad-CAM doğrulaması

| Aşama | Hedef | Ölçülen aktivasyon ve gradyan | Grad-CAM kütüphanesine verilen düzen |
|---|---|---|---|
| 3 | `model.layers[2]` çıkışı | 1 × 14 × 14 × 384 | 1 × 384 × 14 × 14 |
| 4 | `model.layers[3].blocks[-1].norm1` çıkışı | 1 × 7 × 7 × 768 | 1 × 768 × 7 × 7 |

Kurulu Grad-CAM kütüphanesinin kanal ağırlıkları, hedef sınıf logitinin gradyanının uzamsal ortalamasıdır. Aktivasyon kanalları bu ağırlıklarla toplanır ve pozitif katkı ReLU ile tutulur. Haritalar yeniden boyutlandırılır ve kendi ölçeklerinde normalleştirilir. Yatay çevirmeli görünümler ters çevrilerek aynı koordinatlara getirilir. Altı giriş varyantındaki Grad-CAM ortalanır. Görselde ayrıca karanlık piksellerin şeffaf bırakılması kutu hesabındaki ham değerleri etkilemez.

Swin `eval()` modunda doğrulandı. Altı vakada her iki sürümün Grad-CAM hedefi ve ayrı Swin tahmini eşleşti. Te-gl_22 üzerinde iki aşamada gradyan ve aktivasyon boyutları, 224 × 224 çıktı, sonlu değerler, siyah piksellerin şeffaf kalması ve tekrar çalıştırıldığında sayısal tutarlılık kontrol edildi. Her aşamada iki çalıştırmadan sonra hedef katmandaki hook sayısı başlangıç değerine döndü. Bu kontroller bütün görüntülerde anatomik sınır uyumu ölçmez.

## Gerçek örnek sonuçları

| Görüntü | Eski otomatik maske | Güncel otomatik maske | Güncel kutu kaynağı |
|---|---:|---:|---|
| Te-gl_22.jpg | 21.723 piksel | 1.450 piksel | Aşama 3 ve 4 ortak odağı |
| Te-aug-me_7.jpg | 16.967 piksel | 1.059 piksel | Aşama 3 |
| Te-me_263.jpg | 17.781 piksel | 735 piksel | Aşama 3 |
| Te-pi_1.jpg | 19.767 piksel | 3.860 piksel | Aşama 4 |
| Te-gl_1.jpg | Atlandı | Atlandı | İki model de tümör yok tahmini verdi |
| Te-no_1.jpg | Atlandı | Atlandı | Swin tümör yok tahmini verdi |

Te-gl_22’de yalnızca Aşama 3 kullanıldığında aday kutu ön plana göre çok geniş olduğu için reddedildi. Alternatif açıkken iki aşamanın ortak odağından maske üretildi. Te-pi_1’de Aşama 3 aktivasyonları yeterince ayrışmadı ve Aşama 4 kullanıldı. Altı örnekte iki sürümde de dört maske ve iki otomatik atlama görüldü. Güncel sürümde dört maskenin ikisinde başka aşama bilgisi kullanıldı. Bu sayılar örneklem düzeyinde teknik sonuçlardır. Manuel düzenleme ihtiyacını klinik olarak sınıflandıracak referans bulunmadı.

Bu tek çalıştırmada eski yol yaklaşık 0,16–1,46 sn, güncel yol 0,42–1,87 sn sürdü. Süreler modele veri aktarma, Grad-CAM ve gerekiyorsa SAM’ı da içerir. Tek çalıştırma olduğu için gecikme kıyaslaması genellenemez.

Referans segmentasyon maskeleri bulunmadığından Dice, IoU veya klinik doğruluk hesaplanmadı. Daha küçük maskeler alan farkını gösterir. Te-gl_1 sınıflandırma hatası devam ediyor.

## Dosyalar

- [Altı gerçek görüntü için eski ve güncel paneller](gradcam_sam_once_sonra.png)
- [Ölçümler, ağırlık SHA-256 değerleri ve hook doğrulaması](measurements.json)
- Yeniden çalıştırma betiği `experiments/swin_v3/audit_gradcam_sam.py`. Kaynak ZIP’i, eski revizyona erişebilen Git klasörünü ve çıktı klasörünü parametre olarak alır.

## Görsel inceleme ve arayüz testleri

Karşılaştırma panelinde Te-gl_22'nin güncel maskesi MR'daki belirgin bölgenin yalnızca bir kısmını kaplıyor gibi görünüyor. Te-pi_1'de Aşama 4'ten gelen teknik olarak kabul edilmiş maske, orta hattaki hipofiz bölgesinden çok görüntünün yan tarafını kaplıyor. Bu nedenle bu örnek, maske üretiminin başarılı olmasına rağmen konumlandırmanın şüpheli kaldığı bir sonuç olarak gösterilmelidir.

- Üç kutu ve maske seçimi birim testi geçti.
- Gerçek arşivle yerel API duman testi geçti. CNN ve Swin sınıflandırması, iki Grad-CAM aşaması, otomatik ve manuel SAM, tümör yok tahmininde atlama, PNG ve geçersiz giriş doğrulandı.
- Chrome'da 1440 ve 390 piksel genişlikte yatay taşma görülmedi. Varsayılan olarak yalnızca Aşama 3, Aşama 4 seçimi, iki aşamanın karşılaştırılması, alternatif aşamanın açık ve kapalı çalışması, aktarılmış kutunun klavyeyle düzenlenmesi, büyük MR karşılaştırması ve iki saydamlık kontrolü doğrulandı.
