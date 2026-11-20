# Beyin Tümörü Analiz Sistemi · Yerel teslim doğrulaması

## Kaynaklar ve yedek

Görevde adı geçen “Kapsamlı Proje Devir Dokümanı” ve “Profesyonel Web Arayüzü Tasarımı ve Geliştirilmesi” ayrı ekler olarak bulunamadı. Bu çalışmada görev metni, yerel uygulama, Git kopyasındaki CNN notebook'u ve mevcut ağırlıklar kullanıldı. Yedek `.venv/backups/project-20261119-140621/` altında. `.venv` çalışma ortamı ve üretilmiş Python önbellekleri yedeğe dahil edilmedi. Tüm yerel `.pth` dosyalarının SHA-256 değerleri yedekteki karşılıklarıyla eşleşti.

## Aşama sonuçları

| Aşama | Uygulanan ve doğrulanan |
|---|---|
| Model doğruluğu | CNN mimarisi notebook'un 22.9.1 hücresindeki `architecture_code` içeriğinden alındı. Sekiz evrişim ve 3.337.764 parametre, tek kanallı 224 × 224 giriş ve gerçek ön işleme değerleri doğrulandı. Yerel CNN checkpoint'i SHA-256 eşleşmesi, `weights_only=True` ve `strict=True` ile yüklendi. Sınıf sırası glioma, meningioma, notumor, pituitary. SAM ViT-B çalışıyor. Bağımsız ViT sınıflandırıcısı etkin değil. |
| Grad-CAM | CNN hedefi `features.3.block.3`. Swin Aşama 3 hedefi `model.layers[2]`, Aşama 4 hedefi `model.layers[3].blocks[-1].norm1`. Üç regresyon görüntüsünde CNN ve Swin'in hedef sınıfları kendi tahminleriyle eşleşti. |
| SAM ve hizalama | Önceki koordinat iyileştirmesi, doğal en-boy oranı ve açık kullanıcı tercihiyle alternatif aşama korundu. Aynı altı MR karşılaştırması yeni sürümle tekrar çalıştırıldı. Sınıflar, kutular, maske piksel sayıları ve checkpoint hash'leri önceki denetimle eşleşti. Otomatik ve manuel SAM gerçek modelle geçti. |
| Arayüz | Chrome'da 1440 ve 390 piksel, tek/çift aşama, CNN Grad-CAM, maskeler, saydamlık, klavye ile kutu düzenleme, galeri, tema, İngilizce gezinme ve oturum geçmişi doğrulandı. Görüntüleyicide yakınlaştırma, sıfırlama, klavyeyle kaydırma ve tarayıcı içi tam alan doğrulandı. |
| PubMed ve PDF | Gerçek PubMed sorgusu 12 kayıt döndürdü. DOI ve özet doğrulanan örnek PMID 38486342, DOI 10.1186/s40644-024-00682-y. PDF, görüntü hariç ve dahil seçeneklerle açıldı. Aktif Swin checkpoint'i ve model uyumu raporda görüldü. |
| Gizlilik ve sınırlar | Hatalı içerik, tür, boyut, kutu ve parametre reddi duman testinde geçti. Analiz geçmişinin kayıtları yalnızca sayfa belleğinde ve görüntüsüz tutuluyor. PubMed'e MR gönderilmiyor. Klinik doğrulama, hasta düzeyinde ayrım, skor kalibrasyonu ve referans maske ölçümü olmaması belgede ve arayüzde açıklanıyor. |
| Başlatma ve test | `.\start.bat --port 8011` HTTP 200 ile açıldı. Dört birim testi ve gerçek model HTTP duman testi geçti. |

## Açık kalanlar

- Ayrı ekip dağıtım paketi `models/CNN_V2/` altında bulunamadı. Paket yükleme yolu eklendi. Gerçek testte yerel orijinal CNN checkpoint'i kullanıldı.
- Sonraki doğrulamada İngilizce PDF, model uyuşmazlığı metni ve başlıca dinamik API mesajları eklendi. Kalan Türkçe metinler ek denetimde listelenmiştir.
- Yerel CNN grafik klasöründe 29 PNG bulunuyor. Güncel GitHub kaynağından 14_auc_ap_comparison.png tamamlandı. Galeri seçili grafikleri gösteriyor.
- Aktif profil `v3`. Üç profil ayrı sunucularda gerçek MR ile sınandı. Sonuç tablosu ek denetimde.
- Te-gl_1 örneğindeki sınıflandırma hatası ve Te-pi_1 örneğindeki alternatif SAM yerleşim sorunu sürüyor. Referans maskeler olmadan klinik maske doğruluğu, Dice veya IoU raporlanmadı.

## Dosyalar ve çalıştırma

- Model ve API: `backend/cnn_architecture.py`, `backend/models.py`, `backend/segmentation.py`, `backend/main.py`, `backend/research.py`, `backend/report.py`.
- Arayüz: `frontend/index.html`, `frontend/app.js`, `frontend/style.css`.
- Doğrulama ve belgeler: `tests/smoke_web.py`, `README.md`, bu rapor. Mevcut CNN deney grafikleri `results/CNN_V2 sonuçlar/` klasörüne Git kopyasından alındı.
- [Altı gerçek MR'ın önce/sonra paneli](results/swin_v3/gradcam_sam_hizalama/gradcam_sam_once_sonra.png) ve [denetim raporu](results/swin_v3/gradcam_sam_hizalama/README.md).

```powershell
.\start.bat --port 8011
.\.venv\Scripts\python.exe tests/smoke_web.py "C:\veri\Brain_Tumor_MRI_Dataset.zip" --root http://127.0.0.1:8011 --pubmed
```

GitHub'a gönderim yapılmadı. Onaydan sonra güncel depo kopyasında değişen kod, belgeler ve seçili sonuçlar incelenip Türkçe commit hazırlanabilir. Model ağırlıkları, MRI veri seti, yerel yedek ve çalışma ortamı gönderime alınmamalıdır.

[İkinci doğrulama ve ayrıntılı test çıktıları](REVIEW_VERIFICATION.md)

Güncel CNN yükleme kuralı ve 20 MR örneklemi için [son doğrulamaya](FINAL_VERIFICATION.md) bakın.
