# Aşama 3 A varsayılanı doğrulaması

Sunucu ortam değişkeni kaldırılarak 8014 portunda başlatıldı. /health CUDA, Swin v3, stage_output ve layers[2] döndürdü. Alternatif aşama kapalı tutuldu.

| Görüntü | Swin | Durum önceki → güncel | Kutu önceki → güncel | SAM px önceki → güncel |
|---|---|---|---|---|
| Te-gl_22.jpg | glioma | kutu_yok → kutu_yok | — → — | — → — |
| Te-aug-me_7.jpg | meningioma | hazır → hazır | [119, 73, 170, 140] → [119, 73, 170, 140] | 1062 → 1062 |
| Te-gl_1.jpg | notumor | atlandı → atlandı | — → — | — → — |
| Te-me_263.jpg | meningioma | hazır → hazır | [107, 72, 152, 119] → [107, 72, 152, 119] | 748 → 748 |
| Te-pi_1.jpg | pituitary | kutu_yok → kutu_yok | — → — | — → — |
| Te-no_1.jpg | notumor | atlandı → atlandı | — → — | — → — |
| Te-gl_332.jpg | glioma | kutu_yok → kutu_yok | — → — | — → — |
| Te-gl_82.jpg | glioma | hazır → hazır | [76, 38, 136, 86] → [76, 38, 136, 86] | 2670 → 2670 |
| Te-gl_35.jpg | glioma | hazır → hazır | [72, 22, 165, 80] → [72, 22, 165, 80] | 4132 → 4132 |
| Te-gl_87.jpg | meningioma | hazır → hazır | [105, 57, 131, 84] → [105, 57, 131, 84] | 263 → 263 |
| Te-gl_397.jpg | glioma | hazır → hazır | [140, 44, 163, 69] → [140, 44, 163, 69] | 523 → 523 |
| Te-me_106.jpg | meningioma | hazır → hazır | [44, 42, 118, 109] → [44, 42, 118, 109] | 1876 → 1876 |
| Te-aug-me_15.jpg | meningioma | hazır → hazır | [118, 65, 164, 102] → [118, 65, 164, 102] | 1334 → 1334 |
| Te-me_124.jpg | meningioma | hazır → hazır | [73, 56, 115, 101] → [73, 56, 115, 101] | 926 → 926 |
| Te-me_270.jpg | meningioma | hazır → hazır | [101, 37, 138, 73] → [101, 37, 138, 73] | 630 → 630 |
| Te-me_62.jpg | meningioma | hazır → hazır | [65, 76, 102, 115] → [65, 76, 102, 115] | 331 → 331 |
| Te-no_304.jpg | notumor | atlandı → atlandı | — → — | — → — |
| Te-no_267.jpg | notumor | atlandı → atlandı | — → — | — → — |
| Te-no_125.jpg | notumor | atlandı → atlandı | — → — | — → — |
| Te-no_226.jpg | notumor | atlandı → atlandı | — → — | — → — |
| Te-no_380.jpg | notumor | atlandı → atlandı | — → — | — → — |
| Te-pi_122.jpg | pituitary | kutu_yok → kutu_yok | — → — | — → — |
| Te-pi_52.jpg | pituitary | hazır → hazır | [118, 78, 178, 138] → [118, 78, 178, 138] | 2928 → 2928 |
| Te-pi_46.jpg | pituitary | hazır → hazır | [21, 75, 122, 174] → [21, 75, 122, 174] | 7550 → 7550 |
| Te-pi_184.jpg | pituitary | hazır → hazır | [27, 27, 128, 114] → [27, 27, 128, 114] | 6715 → 6715 |
| Te-pi_73.jpg | pituitary | kutu_yok → kutu_yok | — → — | — → — |

26/26 sonuçta tahmin, kutu koordinatları, durum ve maske pikseli birebir eşleşti. Önceki A tablosunda maske hash’i saklanmamıştı. Bu nedenle 14 maske, önceki A sunucusu (8011) ve ortam değişkeni olmadan başlatılan yeni sunucu (8014) üzerinde yeniden alındı. Kodlanmış PNG yerine çözülmüş maske piksel dizilerinin SHA-256 değerleri karşılaştırıldı. [14 çiftin tamamı eşleşti](stage3_a_mask_hashes.csv).
Grad-CAM panelinin canlı HTML alanı ve katman bilgisini /health üzerinden kullanan JavaScript kontrol edildi. Gerçek MR analiziyle oluşturulan PDF’lerde Türkçe ve İngilizce katman metni doğrulandı.

## Testler ve değişiklik kapsamı

- `.venv/Scripts/python.exe -B -m unittest discover -s tests -p 'test*.py'` ile 7 test geçti. Yeni kalıcı testler varsayılanı, block_norm1 seçimini ve geçersiz değeri kapsıyor.
- `.venv/Scripts/python.exe -B tests/smoke_web.py "../Brain_Tumor_MRI_Dataset.zip" --root http://127.0.0.1:8014` geçti.
- Chrome’da gerçek MR ile Grad-CAM başlığı ve açıklaması Türkçe ve İngilizce doğrulandı. PDF metni iki dilde doğrulandı. Bu turda block_norm1 için ayrı tarayıcı koşusu doğrulanmadı.
- Önce yedek alındı. Korunan 44 kaynak/model/test dosyası SHA-256 ile eşleşti. Üretim Python değişikliği models.py içindeki varsayılan değerle sınırlı.
- Değişen kaynaklar backend/models.py, frontend/app.js, frontend/index.html, frontend/messages.json ve README.md. Eklenen test tests/test_swin_stage3_target.py. Eklenen sonuçlar stage3_a_default_validation.md ve stage3_a_mask_hashes.csv. GitHub’a push yapılmadı.
