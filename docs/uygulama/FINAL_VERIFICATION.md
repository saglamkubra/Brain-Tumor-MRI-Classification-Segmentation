# Son yerel doğrulama

Başlangıç yedeği `.venv/backups/final-20261119-232306/` altında. Yedek 398 dosya içeriyor. Çalıştırma komutu `.\start.bat --port 8011`.

## 1. CNN dağıtımı

Komut: `.venv/Scripts/python.exe -X utf8 .venv/final_cnn_policy.py`.

- **Geçti:** Mevcut klasördeki `cnn_v2_best.pth` SHA-256 değeri `ae6501b19f70a1479374d9111cc954a76a2e1fa77a6863ddf159c3b03bf1fa98`. `deployment_config.json` içindeki değerle eşleşti.
- **Eksik:** `models/CNN_V2/cnn_v2_architecture.py` ve `models/CNN_V2/inference.py`. Pakete dosya eklenmedi ve eksik paket kullanılmadı. Bu dosyaların dağıtım kaynağından gelmesi gerekiyor.
- **Geçti:** Gerçek eksik klasör, ortam değişkeni verilse bile reddedildi. Tamamen paketsiz izole testte varsayılan ret ve `NEURO_ALLOW_LOCAL_CNN=1` ile mevcut, sabit hash'e bağlı yerel checkpoint'in kabulü doğrulandı. Bu opt-in yolunda `weights_only=True`, sınırlı NumPy allowlist, `strict=True`, 3.337.764 parametre ve mevcut normalizasyon uygulandı. Tam paket yolu NumPy allowlist kullanmaz. Koddaki gerçek paket normalizasyonu config'ten alınır.
- **Doğrulanmadı:** Tam paketin açılması, bu pakette yanlış config hash'iyle ret, paket üzerinden config normalizasyonunun kullanılması ve `/health` kaynağının `deployment` olması. Bunun için iki eksik dosya gerekli.

## 2. Ortak odak ve 20 ayrı test MR'ı

Komut:

```powershell
..venvScriptspython.exe -m experiments.swin_v3.audit_sam_fallback <MR-arşivi> <V3-test.csv> .venv/backups/review-20261119-142627/backend/segmentation.py results/swin_v3/fallback_validation/v3_sample20.csv
```

Çıkış kodu 0. V3 test listesinden sabit 20261119 tohumuyla her gerçek sınıftan beş benzersiz dosya seçildi. Bilinen regresyon görüntüleri dışarıda bırakıldı. İki sürümde aynı RGB MR, aynı başlangıç V3 checkpoint'i, seçili Aşama 3 ve açık alternatif seçeneği kullanıldı. Eski sürüm uygun ortak odak çıkmazsa diğer aşamanın tek başına önerdiği kutuyu da kullanıyordu. Yeni sürüm o adımı atlayıp manuel kutu öneriyor.

Ortak odak için ön planda her pozitif Grad-CAM haritası kendi zirvesine bölünür. İki haritanın da en az 0,70 olduğu en az 16 önizleme pikseli gerekir. Bu kesişimde geometrik ortalama kullanılır. Mevcut kutu işlevi 3 × 3 ön plan erozyonu ve 7 × 7 Gauss yumuşatma yapar. Zirve ile medyan en az 0,15 ayrışmalıdır. Eşik, 90. yüzdelik ile zirvenin %60'ının büyüğüdür. 8 bağlılıkla en az 10 piksellik bileşenler alınır. İki zirve arasında en az zirvenin %8'i kadar fark gerekir. Kutuya en uzun bileşen kenarının %18'i kadar, 5–12 piksel ile sınırlı kenar boşluğu eklenir. Ön planın %28'ini aşan veya bir kenarı 12 pikselden küçük kutu reddedilir.

Bu sayılar önceki uygulamadan gelen sezgisel eşiklerdir. Klinik veriyle optimize edilmiş eşikler değildir. AST karşılaştırması, Te-pi_1 düzeltmesinden önceki yedek ile şimdiki `shared_cam_focus`, `box_diagnostic` ve `automatic_box` işlevlerinin aynı olduğunu doğruladı. Dolayısıyla bu düzeltme için Te-pi_1'e özel bir eşik değiştirilmedi. Bağımsız örneklem politika değişikliğinin etkisini ölçer. Eşiklerin klinik geçerliliğini doğrulamaz.

| Gerçek sınıf | n | Eski kutu | Yeni kutu |
|---|---:|---:|---:|
| Glioma | 5 | 5 | 4 |
| Meningioma | 5 | 5 | 5 |
| No tumor | 5 | 0 | 0 |
| Pituitary | 5 | 5 | 5 |
| Toplam | 20 | 15 (%75) | 14 (%70) |

Otomatik değerlendirmeye giren 15 örnekte oran %100'den %93,33'e indi. Beş “tümör yok” tahmini iki sürümde de atlandı. Yalnız Te-gl_332 değişti. Önceki tek aşamalı alternatif kutu `[93,190,131,223]` 1.099 piksellik maske verdi. Güncel sürüm kutu döndürmedi. Referans maskeler olmadan bu kararın segmentasyon doğruluğuna etkisi ölçülemez.

[20 vakalık tablo](../../results/swin_v3/fallback_validation/table.md) · [Kutu kaynaklarını da içeren CSV](../../results/swin_v3/fallback_validation/v3_sample20.csv)

## 3. Swin profili

`.venv/Scripts/python.exe .venv/final_http_policy.py` **geçti**. `/health` ve gerçek tahminde profil `v3` ve checkpoint `swin_tiny_v3_best.pth`. PDF metninde ikisi birlikte doğrulandı. CNN eksikken gerçek Swin tahmini, Grad-CAM ve manuel SAM çalıştı. Dört kayıt arayüz ve README'de ayrıdır. Aktif profil değiştirilmedi.

| Deney | Ağırlık kimliği | Doğruluk | Glioma recall |
|---|---|---:|---:|
| İlk V3 | swin_tiny_v3_best.pth | %93,86 | %82,77 |
| Önceki ek eğitim | Önceki koşunun ağırlığı. Dosya kimliği doğrulanmadı. | %94,11 | %81,98 |
| Colab T4 | Colab koşusunun swin_tiny_v3_finetuned_best.pth dışa aktarımı | %93,98 | %81,72 |
| Yerel RTX 3050 | Yereldeki swin_tiny_v3_finetuned_best.pth | %94,05 | %81,98 |

Kayıtlı Te-me_263 deneyinde yerel ince ayar otomatik maskesi 4.396, manuel maske 745 piksel. Bu alan farkı, profil seçiminin otomatik kutuyu etkileyebildiğini gösterir. Segmentasyon doğruluğu ölçümü değildir. Arayüz ve README bunu belirtiyor.

## 4. Koruma ve çalıştırma

- **Geçti:** Geçici Git deposunda `git check-ignore -v` ile `.venv`, `backups`, `.backups`, `yedek`, `models`, `.pth`, MRI ZIP'i ve açılmış `Training`, `Testing`, `dataset` yolları doğrulandı.
- **Geçti:** Dört birim testi (`python -m unittest discover -s tests -p 'test*.py'`), JS sözdizimi ve Python derleme kontrolü.
- GitHub'a push yapılmadı.

Değişenler: `backend/models.py`, `frontend/app.js`, `frontend/index.html`, `frontend/messages.json`, `.gitignore`, `README.md`, `DASHBOARD_VALIDATION.md`, `REVIEW_VERIFICATION.md`. Yeni karşılaştırma betiği `experiments/swin_v3/audit_sam_fallback.py`, CSV/tablo ve bu rapor eklendi. Grad-CAM veya SAM eşikleri bu turda değiştirilmedi.

`.venv/Scripts/python.exe .venv/final_policy_browser.py` **geçti**. Chrome 768 piksel görünümünde eksik CNN uyarısı, aktif `v3` ve checkpoint adı ile dört ayrı kayıt görüldü.
