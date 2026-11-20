# NEURO AI LAB web uygulaması test raporu

18 Kasım 2026 · Windows · NVIDIA GeForce RTX 3050 Laptop GPU (4 GiB)

## Aşama 1 · MR yükleme ve model karşılaştırması

Yerel HTTP üzerinden ana sayfa ve JavaScript yüklendi. `GET /health` üç modelin yüklendiğini bildirdi. `POST /predict` isteğine `Testing/meningioma/Te-me_263.jpg` yüklenince CNN, ViT ve Swin-Tiny aynı sınıf sırasıyla dört gerçek olasılık döndürdü. Her modelin olasılıklarının toplamı testte 1 olarak doğrulandı. Üçünün de en yüksek olasılığı `meningioma` idi.

| Model | CUDA meningioma | CPU meningioma |
| --- | ---: | ---: |
| CNN | %84,45063 | %84,45224 |
| ViT | %98,98254 | %98,98257 |
| Swin-Tiny | %99,99818 | %99,99818 |

## Aşama 2 · Grad-CAM, SAM ve hata durumları

Tam uygulama CUDA 12.4 destekli PyTorch 2.6.0 ile RTX 3050 üzerinde ve resmi PyTorch 2.6.0 CPU paketiyle ayrı ayrı HTTP üzerinden test edildi. `GET /health` dört yüklenmiş model bildirdi. Hem PNG hem JPG tahminleri çalıştı. Grad-CAM ve SAM PNG çıktıları çözümlenerek boyutları doğrulandı.

| SAM yöntemi | Kutu | CUDA kalite skoru | CPU kalite skoru | Maske alanı |
| --- | --- | ---: | ---: | ---: |
| Swin Grad-CAM ile otomatik | [25, 22, 198, 202] | 0,976935 | 0,976930 | 27.787 piksel |
| Manuel | [115, 85, 160, 135] | 0,935452 | 0,935443 | 732 piksel |

Bu örnekte otomatik maske görüntünün yaklaşık %55’ini kapladı. SAM kalite skoru, maskenin tümör sınırını doğru gösterdiğini doğrulamaz.

- Gerçek `notumor` örneğinde Grad-CAM üretildi ve otomatik SAM atlandı.
- Geçersiz manuel kutu ve bozuk görüntü 422, MIME uyumsuzluğu 415, bayt veya piksel sınırı aşımı 413 döndürdü.
- Tamamen sıfır Grad-CAM girdisinin otomatik kutu üretmediği doğrudan test edildi. Gerçek bir MR ile kutu bulunamaması arayüzde yeniden üretilmedi.
- Enjekte edilen CUDA bellek hatası sonrası CNN'in CPU'da çıkarıma devam ettiği ve CUDA'daki ayrılmış tensör belleğinin sıfırlandığı doğrulandı. Gerçek VRAM taşması oluşturulmadı.
- CPU doğrulamasında CUDA paketini `CUDA_VISIBLE_DEVICES=-1` ile çalıştırma denemesi ilk tahminde süreci sonlandırdı. Yukarıdaki CPU sonuçları, ayrı bir ortamda resmi CPU paketiyle başarıyla tamamlandı.

## Windows ve tarayıcı

`start.bat` mevcut proje sanal ortamıyla çalıştırıldı. Sunucu `127.0.0.1:8000` üzerinde açıldı ve Microsoft Edge otomatik olarak uygulamaya gitti. Edge'de gerçek JPG seçildi, üç tahmin kartı ve Grad-CAM görüntülendi. Fareyle manuel kutu çizildi ve manuel maske üretildi. Otomatik ve manuel maskeler aynı karşılaştırma alanında görüntülendi. İlk bağımlılık kurulumu boş bir sanal ortamdan `start.bat` ile yeniden denenmedi.

## Güncel platform doğrulaması

- `start.bat --port 8011` ile Windows sunucusu başlatıldı. CUDA üzerinde CNN V2, Swin-Tiny V3 ve SAM ViT-B yüklendi.
- Dört örnek JPG ve bir PNG ile canlı sınıflandırma doğrulandı. Meningioma, hipofiz tümörü ve tümör yok örneklerinde iki sınıflandırıcı da doğru sınıfı seçti. Kullanılan glioma örneğinde ikisi de tümör yok tahmini verdi.
- Aynı meningioma örneğinde aşama 3 ve 4 Grad-CAM haritaları farklı çıktı verdi. Otomatik ve manuel SAM 224 × 224 boş olmayan maskeler üretti. Bunlar işlev testleridir. Tümör sınırı doğruluğu ölçülmedi.
- Tümör yok sınıfında otomatik SAM atlandı. Hatalı kutu, bozuk içerik, yanlış dosya türü ve boyut sınırı denetimleri geçti.
- Canlı PubMed sorgusu 12 kayıt döndürdü. PDF raporu tahminler, orijinal görüntü, iki Grad-CAM çıktısı, SAM çıktıları ve gerçek PubMed bağlantıları ile üç sayfa üretildi.
- Chrome, 1440 ve 390 piksel genişliklerde iki performans kartını yatay taşma olmadan çizdi. Demo tahmini, otomatik SAM ve iki Grad-CAM çıktısı tarayıcıda çalıştı.
- Aynı 1.579 görüntülük V3 listesinde CNN V2 doğruluğu %89,61 ve makro F1 %89,55, Swin-Tiny V3 doğruluğu %93,86 ve makro F1 %93,79. Ayrıntılar `results/web_v3_performance.json` dosyasında. Hasta düzeyinde bağımsızlık doğrulanamadı.
