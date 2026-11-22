# NEURO AI LAB · Veri sızıntısı ve tekrar denetimi

**Kapsam:** Yerel Brain Tumor MRI Dataset arşivindeki 7.200 görüntü, üç notebook’un kayıtlı kodu ve çıktıları, önceki değerlendirmenin tahmin CSV’si. Denetim sırasında kaynak veriler salt okunur açıldı. Model eğitimi veya çıkarımı çalıştırılmadı. Yalnızca bu rapor ve denetim çıktıları üretildi.

**Sonuç:** Arşiv sınırları arasında **kesin görüntü tekrarı var**. Testin 114 görüntüsünün (%7,13) `Training/` altında aynı boyutta, piksel piksele aynı RGB karşılığı bulunuyor. Bu 114 görüntünün özgün JPEG baytları farklı. Notebook’taki eğitim/doğrulama bölmesi yeniden oluşturulduğunda bu iki küme arasında **78 bayt-eş çift** ve **10 ek piksel-eş çift** bulundu. Özgün CSV’ler ve güvenilir hasta kimlikleri yerelde yok. Bu nedenle checkpoint’lerin geçmişte kullandığı satırlar ile hasta düzeyinde sızıntı kesinleştirilemedi.

## Veri bölmeleri nasıl oluşturuldu?

- **Swin**, sıfır tabanlı hücre 6’da `Training/` klasöründeki görüntüleri sınıf sırasıyla, her sınıfta dosya adına göre sıralıyor. `train_test_split(test_size=0.20, random_state=42, stratify=df["sinif"])` ile görüntü düzeyinde ayırıyor. Hücre 7, listeleri `train.csv` ve `val.csv` olarak yazıyor, `Testing/` altındaki bütün uygun görüntüleri `test.csv` olarak kaydediyor. Hücreler 10 ve 12 bu CSV’leri tekrar okuyor. [Swin notebook](../../notebooks/Swin_Transformer_SAM.ipynb)
- CNN notebook’u Google Drive’daki `veri_bolunmesi/train.csv`, `val.csv` ve `test.csv` listelerini okuyor. Kaydedilen boyutlar 4.480/1.120/1.600. [CNN notebook](../../notebooks/CNN.ipynb)
- Özgün CSV’ler çalışma alanında, üst klasörün aranabilen dosyalarında veya yereldeki GitHub geçmişi kopyasında bulunmadı. Arşivde yalnızca görüntü dosyaları var. Dolayısıyla üç notebook’un **özgün satır üyeliği** tek tek karşılaştırılamadı. Aşağıdaki eğitim/doğrulama ayrımı Swin kodunun yerel arşiv üzerinde yeniden üretilmiş halidir. Test ile birleşik `Training/` arasındaki sonuç bu yeniden bölme varsayımından bağımsızdır.

| Bölme | Toplam | Sınıf başına |
| --- | ---: | ---: |
| Eğitim | 4.480 | 1.120 |
| Doğrulama | 1.120 | 280 |
| Test | 1.600 | 400 |

Yerel iki ZIP kopyasının SHA‑256 özeti aynıydı. Denetlenen arşivin özeti `882817250048c78ef7a759cf23e540d7b581f2327b16663c9d3db12f5d2ffdb4`.

## Kesin tekrarlar

Her dosya iki şekilde özetlendi. Özgün baytların SHA‑256 özeti dosyanın birebir kopyalarını buldu. Boyut ve çözümlenmiş RGB piksel baytlarının SHA‑256 özeti, dosya baytları farklı olsa bile aynı çözünürlükte piksel piksele eşleşen görüntüleri buldu. Tablodaki ilk iki sayı birbirini dışlar. Çift sayısı dosya sayısından farklıdır çünkü bir dosya birden fazla kopyayla eşleşebilir.

| Karşılaştırma | Aynı dosya baytları | Yalnız çözümlenmiş RGB eş | Toplam kesin çift | Kesin çiftlere katılan benzersiz dosya |
| --- | ---: | ---: | ---: | ---: |
| Eğitim ↔ doğrulama | 78 | 10 | 88 | 148 |
| Eğitim ↔ test | 0 | 94 | 94 | 181 |
| Doğrulama ↔ test | 0 | 27 | 27 | 52 |
| Eğitim içinde | 122 | 19 | 141 | 216 |
| Doğrulama içinde | 10 | 3 | 13 | 26 |
| Test içinde | 17 | 5 | 22 | 41 |

- Küme sınırlarını aşan **209 kesin çift** toplam **372 benzersiz dosyaya** dokunuyor. Tüm karşılaştırmalarda **385 kesin çift** var. Kesin çiftlerin tamamında dosya adları farklı, sınıf etiketleri ise aynı.
- Eğitim/doğrulama tarafında piksel eşi bulunan **114 benzersiz test görüntüsü** meningioma (100), notumor (9) ve glioma (5) sınıflarından. Pituitary test görüntülerinde karşı eş bulunmadı.
- Eğitim ↔ doğrulama arasındaki 88 kesin çift notumor (57), pituitary (25) ve meningioma (6) sınıflarında. Testin kendi içindeki 22 kesin çift glioma (18) ve meningioma (4) sınıflarında.

## pHash ve görsel inceleme

Görüntüler gri tonlamalı 32×32 önizlemeye dönüştürüldü. İki boyutlu DCT’nin düşük frekanslı 8×8 bölgesinden pHash üretildi. DC katsayısı medyan hesabından çıkarıldı. Kesin SHA‑256 eşleri hariç tutularak Hamming uzaklığı ≤4 olanlar tarandı. Standart boyuttaki iki gri görüntünün (64×64) ortalama mutlak piksel farkı ≤5/255 ise çift görsel inceleme adayı olarak kaydedildi. Eşikler sıkıştırma ve hafif dönüşüm adayları bulmak için kullanıldı. Kırpılmış, döndürülmüş veya belirgin kontrast değişimi içeren tekrarlar kaçabilir.

| Karşılaştırma | Yalnız pHash eşiğini geçen çift | Ek piksel farkı filtresini de geçen aday çift |
| --- | ---: | ---: |
| Eğitim ↔ doğrulama | 1.055 | 445 |
| Eğitim ↔ test | 1.515 | 558 |
| Doğrulama ↔ test | 409 | 178 |
| Eğitim içinde | 2.012 | 741 |
| Doğrulama içinde | 139 | 59 |
| Test içinde | 401 | 167 |

- Görsel inceleme listesi **2.148 aday çift** içeriyor. Bunların **1.181’i** bölüm sınırlarını aşıyor ve 1.078 benzersiz dosyaya dokunuyor. Bölümler arası adayların 1.104’ü notumor ↔ notumor, 34’ü meningioma ↔ meningioma, 31’i glioma ↔ glioma, 8’i pituitary ↔ pituitary ve 4’ü meningioma ↔ glioma.
- Eğitim/doğrulama ile aday eşi bulunan **338 benzersiz test görüntüsü** var. Bunların 15’i başka bir çift üzerinden kesin tekrar listesinde de var. Geriye **323 ek test adayı** kalıyor. Bunlar notumor (307), glioma (11) ve meningioma (5).
- **27 çifti yan yana görsel olarak inceledim.** Buna farklı etiketli sekiz adayın tamamı ve üç bölüm karşılaştırmasından 19 aynı etiketli örnek dahil. İncelenen çiftlerin 18’i aynı kesitin varyantı olabileceğine dair güçlü görsel benzerlik gösterdi, 8’i belirsiz kaldı, 1’inde anatomik görünüm belirgin biçimde farklıydı. Kalan **2.121 çift** görsel olarak tek tek incelenmedi. Karşılaştırma dosyası her bir adayın durumunu gösteriyor.
- Sekiz farklı etiketli adayın dördü eğitim ↔ test, dördü test içi. Görsel olarak iki grupta toplanıyorlar. `Te-gl_68.jpg` ve `Te-gl_160.jpg` glioma etiketli, yakın karşılıkları ise meningioma etiketli. Sekiz çiftte anatomik yapı ve lezyon görünümü çok yakın. Görsel inceleme, hangi etiketin doğru olduğunu veya aynı hastaya ait olduklarını belirlemiyor. Kesin bayt/piksel eşlerinde farklı etiket saptanmadı.

Görsel karşılaştırma dosyası yerel çalışma arşivinde. [Kesin ve aday çiftlerin CSV kaydı](../../data_leakage_audit/matches.csv)

## Hasta düzeyi, dönüşümler ve model seçimi

- **Hasta düzeyinde bağımsızlık doğrulanamadı.** Yerel arşiv 7.200 JPG görüntüden oluşuyor. Hasta veya çalışma kimliğine bağlanabilen bir manifest yok. JPEG’lerin 22’sinde EXIF, 5’inde yorum alanı var. İncelenen meta veri alanları bölme için kullanılabilir güvenilir bir hasta/çalışma anahtarı sağlamadı. Kimlik veya kişisel meta veri değerleri denetim çıktılarına aktarılmadı. Görüntü düzeyindeki bölme kodunda hasta gruplaması bulunmuyor. Aynı hastanın farklı kesitleri kümeler arasında dağılmış olabilir. Veri kümesinin birden çok koleksiyondan derlendiği, küratörün yayımladığı açıklamada belirtiliyor. [Veri kümesi açıklaması](https://www.kaggle.com/dsv/2645886)
- CNN sabit gri tonlama, boyutlandırma ve 0,5 normalizasyonu kullanıyor. Swin döndürmeyi yalnız eğitimde, sabit ImageNet normalizasyonunu doğrulama ve testte uyguluyor.
- **Arşivde hazır varyant riski ayrıca var.** `aug` geçen dosya adları eğitimde 80, doğrulamada 20 ve testte 103 kez görülüyor. Hepsi meningioma. Dosya adı hangi dönüşümün uygulandığını kanıtlamıyor. Notebook içindeki rastgele artırmaların doğru ayrılmış olması, arşivle gelen olası varyantların kümeler arası dağılımını engellemiyor.
- **Sınıflandırıcı checkpoint seçimi doğrulama ile yapılıyor.** CNN 20 epoch boyunca en düşük validation loss’u, Swin 3 epoch boyunca en yüksek validation accuracy’yi kaydediyor. Kayıtlı döngüler sabit epoch sayısıyla çalışıyor ve durdurma koşulu içermiyor. Test loader checkpoint seçim ölçütüne veya eğitim döngüsüne bağlı görünmüyor. Notebook dışındaki önceki denemelerin ya da kararların geçmişi doğrulanamadı.
- **Grad-CAM/SAM akışında testten örnek bazında ayar geliştirme var.** Swin notebook’u etiketli test örneklerinde Grad-CAM hedef katmanlarını ve otomatik kutu eşiklerini deniyor, ardından aynı test listesinde örnek çıktılar üretiyor. Bu, sınıflandırıcı checkpoint seçimiyle ayrı bir bulgu. Bu örnekler yöntemin geliştirilmiş örnek çıktıları olarak yorumlanmalı.

## Performansa olası etkisi

CNN ve Swin-Tiny için saklanan 3.200 tahmin satırı kullanılarak alt kümeler yeniden hesaplandı. Yeni çıkarım çalıştırılmadı. Önceki performans raporunun benzerlik taraması dHash kullanıyordu. Bu tabloda yeni pHash taraması kullanıldığı için son alt kümenin büyüklüğü farklı.

| Test alt kümesi | Görüntü | CNN accuracy / macro F1 | Swin-Tiny accuracy / macro F1 |
| --- | ---: | ---: | ---: |
| Tüm test | 1.600 | %76,75 / %75,88 | %93,25 / %93,19 |
| Eğitim/doğrulamada piksel eşi bulunanları hariç tut | 1.486 | %81,36 / %79,99 | %93,00 / %92,73 |
| Eğitim/doğrulama ile pHash adaylarını da hariç tut | 1.163 | %76,61 / %73,15 | %91,23 / %90,25 |

Model sıralaması iki alt kümede de korundu. Piksel-eş kayıtları çıkarmak Swin-Tiny accuracy’sini 0,25 puan düşürürken CNN accuracy’sini artırdı. Sınıf bileşimi de değişti. Son alt kümede 400 notumor görüntüsünden 84’ü kaldı. Bu farklar veri sızıntısının nedensel etkisini ölçmez. Aynı veya ilişkili kesitlerin eğitime girmiş olması değerlendirmeyi gereğinden iyimser yapabilir, test içi tekrarlar da bazı kesitlere fazladan ağırlık verir. Ayrıca etiket uyuşmazlığı, yüksek skorlu bazı yanlışların yorumunu değiştiriyor. `Te-gl_68.jpg` Swin-Tiny’nin önceki testteki en yüksek skorlu yanlışıydı. [Önceki değerlendirme](../deneyler/MODEL_PERFORMANCE_ANALYSIS.md)

## Önerilen düzeltmeler

1. Orijinal CSV’leri ve checkpoint–veri sürümü eşlemesini koru. Gelecekteki listelerde görüntü SHA‑256 özeti, kaynak koleksiyon ve doğrulanmış anonim grup kimliği bulunsun.
2. Kesin tekrarları ayırma öncesinde gruplandır. Görsel adayları incele ve özellikle iki çelişkili glioma/meningioma grubunu kaynak etiketlerle doğrula. Aynı görüntünün varyantlarını aynı bölmede tut.
3. Güvenilir hasta kimliği bulunabilen kaynaklarda bölmeleri hasta düzeyinde yeniden kur. Kimliği doğrulanamayan görüntüleri ayrı işaretle ve mümkünse bağımsız bir değerlendirme kümesi oluştur.
4. Bölme tamamlandıktan sonra eğitim artırmalarını uygula. Öğrenilen dönüşüm parametreleri varsa yalnızca eğitimden hesapla. Model, eşik ve Grad-CAM/SAM ayarlarını doğrulamada seç ve nihai testi sabit tut.
5. Yeni ve bağımsız bölmede mevcut checkpoint’leri tekrar ölç. Yeniden eğitim kararını bu ölçüm ve etiket denetimi sonrasında ver.
