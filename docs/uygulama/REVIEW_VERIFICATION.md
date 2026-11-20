# Yerel teslim · İkinci doğrulama

## CNN checkpoint ve dağıtım paketi

`.venv/Scripts/python.exe .venv/review_cnn_integrity.py` geçti. Sunucu `models/cnn_v2_best.pth` dosyasını kodda sabitlenen yerel referans SHA-256 değeri `ae6501b19f70a1479374d9111cc954a76a2e1fa77a6863ddf159c3b03bf1fa98` ile doğruladı. Bu değer mevcut yerel dosyanın sabit referansıdır. Drive paketinin yayımladığı hash henüz doğrulanmadı. Model 3.337.764 parametreyle, sınıf sırası glioma, meningioma, notumor, pituitary ve normalizasyon 0,17666279376003613 / 0,19102925562565193 ile açıldı.

[Model yükleyicisi](../../backend/models.py) gerçek çağrıyı sınırlı NumPy allowlist içinde `torch.load(..., weights_only=True)` olarak yapıyor. Hash önce kontrol ediliyor. HTTP çıktısında CNN profili `v2`, kaynak `pinned_local`, `weights_only:true`, `numpy_allowlist:true` görüldü. Arayüz CNN paketinin yerine bu doğrulanmış yerel dosyanın kullanıldığını belirtiyor. Dağıtım klasörü varsa uyumsuz paket yüklenmez. Doğru hash kabulü ve yanlış hash'in PyTorch yüklemesinden önce reddi, gerçek yerel CNN tensörleriyle hazırlanmış geçici test paketinde doğrulandı.

Drive'da `Beyin_Tumoru_Projesi/deployment/CNN_V2/` ve paket arşivi görüldü. İndirme başlatıldı, ancak tarayıcı erişim hatasından sonra yerelde tamamlanan arşiv bulunmadı. Drive paketinin kendisini sınamak için şu dört dosya `models/CNN_V2/` altına konmalıdır: `cnn_v2_best.pth`, `deployment_config.json`, `cnn_v2_architecture.py`, `inference.py`. Bu paketin SHA-256 eşleşmesi ve gerçek paket normalizasyonu doğrulanmadı.

## Grafikler ve Swin profilleri

GitHub `origin/main` revizyonu 6c645ae içindeki `14_auc_ap_comparison.png` yerel klasöre tamamlandı. Toplam 29 CNN PNG var. `.venv/Scripts/python.exe .venv/current_review_api.py <MR-arşivi>` ile 29 CNN dosyasının ve seçili diğer galeri bağlantılarının HTTP üzerinden PNG döndürdüğü doğrulandı. Toplam 36 benzersiz bağlantı geçti.

Git 46847fa sürümünün varsayılanı `legacy`. Mevcut yerel kodun varsayılanı önceki değişikliklerden beri `v3`. Canlı çalıştırmada ek `NEURO_SWIN_PROFILE` değeri verilmedi. Bu görev sunum için yeni bir varsayılan belirlemedi. Arayüz ve iki dilde PDF aktif checkpoint adını gösteriyor.

`.venv/Scripts/python.exe .venv/review_profiles.py <MR-arşivi>` her profili ayrı 8014 portu sunucusunda başlattı. Test MR'ı Te-me_263.

| Profil | Checkpoint | Sağlık / tahmin | Grad-CAM | Otomatik SAM | Manuel SAM |
|---|---|---|---|---|---|
| legacy | swin_tiny_best.pth | Hazır, meningioma | Aşama 3 ve 4 geçti | Kutu yok; maske zorlanmadı | 745 piksel |
| v3 | swin_tiny_v3_best.pth | Hazır, meningioma | Aşama 3 ve 4 geçti | Aşama 3, 748 piksel | 745 piksel |
| v3-finetuned | swin_tiny_v3_finetuned_best.pth | Hazır, meningioma | Aşama 3 ve 4 geçti | Aşama 3, 4.396 piksel | 745 piksel |

Başlangıç V3 %93,86 doğruluk ve %82,77 glioma recall verdi. Önceki ek eğitim kaydı %94,11 / %81,98, Colab T4 %93,98 / %81,72. Yereldeki fine-tuned dosyanın kayıtlı RTX ölçümü %94,05 / %81,98. Bunlar farklı deney kayıtlarıdır. Bu duman testi, profillerin genel performansını yeniden ölçmedi.

## Te-pi_1 ve Te-gl_1

`.venv/Scripts/python.exe .venv/review_pituitary_reason.py <MR-arşivi>` geçti. Te-pi_1'de Aşama 4 elle seçildiğinde önceki otomatik alternatif ile aynı `[16,88,91,186]` kutusu ve 3.774 piksellik maske yeniden üretildi. Yerel orijinal kutudan önizlemeye ters dönüşüm eşleşti. Bulgular, bu örnekte sorunun kutu koordinat ofseti yerine alternatif aşamanın seçtiği yanal adayla ilişkili olduğunu gösteriyor. Alternatif otomatik yol artık iki aşama arasında güçlü ortak odak arıyor. Ortak odak yoksa manuel kutu öneriliyor. Sayısal eşiklerde bu örneğe özel ayar yapılmadı. Elle Aşama 4 seçilirse aynı aday hâlâ üretilebilir.

`.venv/Scripts/python.exe -m experiments.swin_v3.audit_gradcam_sam <MR-arşivi> --repository .venv/github-results-sync --output results/swin_v3/review_verification` geçti. Aşağıdaki eski sütun, bir önceki yerel koordinat denetiminin alternatif açık sonuçlarıdır. Yeni sütun aynı koşulları gösterir.

| Örnek | Eski kutu / piksel | Yeni kutu / piksel |
|---|---|---|
| Te-gl_22 | [63,92,97,136] / 1.467 | [63,92,97,136] / 1.467 |
| Te-aug-me_7 | [119,73,170,140] / 1.062 | [119,73,170,140] / 1.062 |
| Te-gl_1 | Atlandı | Atlandı |
| Te-me_263 | [107,72,152,119] / 748 | [107,72,152,119] / 748 |
| Te-pi_1 | [16,88,91,186] / 3.774 | Kutu yok |
| Te-no_1 | Atlandı | Atlandı |

Te-gl_1 arşivde glioma etiketli. Başlangıç V3 ve yerel ince ayar Swin dosyası tümör yok, legacy meningioma tahmini verdi. CNN tümör yok tahmini verdi. Sınıf sırası, checkpoint kimliği ve Swin RGB ImageNet normalizasyonu korundu. Bilinen sınıflandırma hatası README ve uygulamanın sınırlamalarına yazıldı. Model yeniden eğitilmedi ve referans segmentasyon doğruluğu ölçülmedi.

## Arayüz, dil ve diğer testler

- `.venv/Scripts/python.exe .venv/review_browser_runner.py` geçti. 768 pikselde yatay taşma yok. Te-gl_102 ile gerçek CNN/Swin ayrışması Türkçe mesajı tetikledi. Aşama 3 varsayılan, açıkça etkinleştirilen alternatif ayrı, Aşama 4 görünümü kendiliğinden açılmadı. İngilizce kutu yok mesajı ve kaynak doğrulandı.
- `.venv/Scripts/python.exe .venv/native_api_qa.py <MR-arşivi>` geçti. Te-gl_1 (354 × 442) ve Te-gl_103 (541 × 477) oranları korundu. Dikey MR'da gerçek manuel maske, yerel kutu ve boş alan hizalaması doğrulandı.
- `.venv/Scripts/python.exe .venv/current_review_api.py <MR-arşivi>` geçti. 422 bozuk dosya, 413 10 MiB ve 413 görüntü ölçüsü sınırları, varsayılan aşama ve iki dilde PDF test edildi. İlk PDF denemesi bir isim çakışmasıyla 500 verdi. Kod düzeltilip Türkçe ve İngilizce PDF tekrar doğrulandı.
- `tests/smoke_web.py ... --pubmed` geçti. Gerçek PubMed özeti, CNN/Swin çıkarımı, Grad-CAM, SAM, PDF ve dosya sınırları kontrol edildi. Dört birim testi geçti.
- Ortak çeviri dosyası `frontend/messages.json`. Dinamik arayüz ve HTTP yanıtları ile PDF başlıkları, sınıf adları, uyum ve akademik uyarılar bu sözlüğü kullanır. Henüz tam çevrilmeyenler: ham PyTorch/checkpoint hata ayrıntıları, bazı işlem sonrası metrik ve SAM açıklamaları, arşiv grafiklerinin başlık/açıklamaları ve önceden üretilmiş Türkçe rapor notları. Gerçek PubMed makaleleri ve grafik içindeki kaynak metinleri kendi dillerinde kalır.
- CNN ve Swin tahmin/sağlık akışında yer alır. Bağımsız sınıflandırıcı kodu, görünümü, testi ve README anlatımı bulunmaz. SAM ViT-B korunur. Arşiv checkpoint'lerinin SHA-256 değerleri değişmedi. Model Hatası İnceleme bölümü eklenmedi. Sahte sonuç üretilmedi. GitHub'a gönderim yapılmadı ve .pth dosyaları ile MR arşivi yerelde kalır.

Değişen dosyalar: `backend/main.py`, `backend/models.py`, `backend/segmentation.py`, `backend/report.py`, `backend/language.py`, `frontend/app.js`, `frontend/index.html`, `frontend/messages.json`, `tests/smoke_web.py`, `README.md`, `DASHBOARD_VALIDATION.md`. Yeni CNN grafiği ve doğrulama çıktıları results altında. Başlangıç yedeği `.venv/backups/review-20261119-142627/`.

Çalıştırma: `.\start.bat --port 8011`. Bir deneme için PowerShell'de `$env:NEURO_SWIN_PROFILE = 'legacy'` veya `'v3-finetuned'` ayarlanabilir. Yerel varsayılana dönmek için `Remove-Item Env:NEURO_SWIN_PROFILE` ve sunucuyu yeniden başlatın.

Güncel CNN yükleme kuralı ve 20 MR örneklemi için [son doğrulamaya](FINAL_VERIFICATION.md) bakın.
