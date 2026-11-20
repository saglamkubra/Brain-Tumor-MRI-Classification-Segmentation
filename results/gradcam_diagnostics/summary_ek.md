# Grad-CAM ek tanı çalışması

Başlangıçta proje yedeği alındı. GitHub’a push yapılmadı. Önceki 40 görüntü CSV listesinden alındı ve çözümlenen RGB piksel SHA-256 değerleri tek tek eşleşti. Önceki model tahminleri sabit hedef olarak yeniden doğrulandı.

## Komut ve çıktı özeti

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
.venv/Scripts/python.exe -B -u .venv/backups/gradcam-ek-20261120/run_ek.py
.venv/Scripts/python.exe -B .venv/backups/gradcam-ek-20261120/report_ek.py
```

g komutu 480 harita karşılaştırması ve 240 kutu kaydı, h üç çift için 120 IoU kaydı, j iki model × 40 görüntü × dokuz müdahale ile 720 skor kaydı üretti. Rapor komutu kayıt sayıları ve hash eşitliğini kontrol etti. Ham sonuçlar [metrics_ek.csv](metrics_ek.csv) dosyasındadır.

## g. Notebook ile uygulama karşılaştırması

Notebook dizinleri sıfır tabanlıdır. CNN V2 hücre 74’ün sınıfı proje kaynak kopyasındaki CNN_V2.ipynb dosyasından alındı. Swin hücre 16 ve 19’un reshape işlevleri notebook’tan alındı ve aynı hedef katmanlarda notebook’ların varsayılan yumuşatmasız Grad-CAM çağrısı kullanıldı. Hücrelerin örnek seçme ve çizim bölümleri devreye alınmadı. Modeller, ön işleme ve önizleme geometrisi üretimden kullanıldı. Bölge müdahalesi, logit ve harita karşılaştırmalarında ikinci bir checkpoint kullanılmadı.

| Profil | Kontrollü karşılaştırma | Spearman (önizleme) | SSIM (önizleme) | Spearman (girdi) | SSIM (girdi) |
|---|---|---:|---:|---:|---:|
| CNN | Notebook / üretim | 0.682 (n=39) | 0.711 (n=40) | 0.680 (n=39) | 0.707 (n=40) |
| CNN | Notebook / üretim yumuşatma kapalı | 1.000 (n=39) | 1.000 (n=40) | 1.000 (n=39) | 1.000 (n=40) |
| CNN | Üretim yumuşatma kapalı / açık | 0.682 (n=39) | 0.711 (n=40) | 0.680 (n=39) | 0.707 (n=40) |
| CNN | Notebook katmanı, kütüphane ve yumuşatma açık / üretim | 1.000 (n=40) | 1.000 (n=40) | 1.000 (n=40) | 1.000 (n=40) |
| Swin3 | Notebook / üretim | 0.484 (n=40) | 0.374 (n=40) | 0.454 (n=40) | 0.356 (n=40) |
| Swin3 | Notebook / üretim yumuşatma kapalı | 0.493 (n=40) | 0.349 (n=40) | 0.465 (n=40) | 0.331 (n=40) |
| Swin3 | Üretim yumuşatma kapalı / açık | 0.929 (n=40) | 0.835 (n=40) | 0.926 (n=40) | 0.833 (n=40) |
| Swin3 | Notebook katmanı, kütüphane ve yumuşatma açık / üretim | 0.514 (n=40) | 0.404 (n=40) | 0.483 (n=40) | 0.386 (n=40) |
| Swin4 | Notebook / üretim | 0.936 (n=40) | 0.897 (n=40) | 0.933 (n=40) | 0.896 (n=40) |
| Swin4 | Notebook / üretim yumuşatma kapalı | 1.000 (n=40) | 1.000 (n=40) | 1.000 (n=40) | 1.000 (n=40) |
| Swin4 | Üretim yumuşatma kapalı / açık | 0.936 (n=40) | 0.897 (n=40) | 0.933 (n=40) | 0.896 (n=40) |
| Swin4 | Notebook katmanı, kütüphane ve yumuşatma açık / üretim | 1.000 (n=40) | 1.000 (n=40) | 1.000 (n=40) | 1.000 (n=40) |

CNN’de iki uygulama aynı konvolüsyonu ve hedef logiti kullanıyor, kanal ağırlıkları ortalama gradyandan geliyor ve ikisinde de pozitif ReLU var. Elle yazılan sınıf ileri geçiş hook’u, autograd.grad ve PyTorch interpolasyonu kullanıp yeniden boyutlandırma sonrasında normalize ediyor. Kütüphane aktivasyon ve gradyanı kendi hook’larıyla alıyor, OpenCV ölçekleme ve kendi normalizasyon yolunu kullanıyor. Yumuşatmasız karşılaştırma bu farkların birleşimini ölçer. Hook, interpolasyon, ReLU ve normalizasyonun CNN’deki bireysel katkıları ayrı ayrı doğrulanmadı.

Swin 4’te aynı norm katmanı kullanıldı. Swin 3’te notebook son bloğun norm1 çıktısını, üretim ise aşama çıkışını kullanıyor. Kontrol satırları yumuşatmanın etkisini ve aynı kütüphane/yumuşatma ile katman farkını ayrı gösterir. Ön işleme her karşılaştırmada aynı tensördü. Önizleme projeksiyonu karşılaştırmanın iki tarafına da uygulandı.

## h. Üretim kutu kuralı

Tüm haritalara mevcut box_diagnostic ve aynı kafa ön planı uygulandı. Koordinatlar ve ret gerekçeleri CSV’de. Alan 224×224 önizlemede iki uç dahil piksel sayısıdır. Alan ortalaması kabul edilen kutularda, IoU ortalaması iki kutu da üretildiğinde hesaplandı. Tek kutu üretilen ve hiç kutu üretilmeyen görüntüler IoU’ya sıfır olarak katılmadı.

| Gerçek sınıf | Harita | Notebook kutusu / n | Notebook alanı | Üretim kutusu / n | Üretim alanı |
|---|---|---:|---:|---:|---:|
| glioma | CNN | 7/10 | 1896.286 (n=7) | 6/10 | 2763.500 (n=6) |
| glioma | Swin3 | 8/10 | 2517.250 (n=8) | 4/10 | 1851.250 (n=4) |
| glioma | Swin4 | 4/10 | 3571.500 (n=4) | 5/10 | 4551.800 (n=5) |
| meningioma | CNN | 7/10 | 2338.857 (n=7) | 5/10 | 1974.800 (n=5) |
| meningioma | Swin3 | 7/10 | 2238.429 (n=7) | 7/10 | 2855.143 (n=7) |
| meningioma | Swin4 | 6/10 | 6481.667 (n=6) | 5/10 | 6515.600 (n=5) |
| notumor | CNN | 9/10 | 2681.667 (n=9) | 5/10 | 1257.800 (n=5) |
| notumor | Swin3 | 5/10 | 2432.000 (n=5) | 5/10 | 985.800 (n=5) |
| notumor | Swin4 | 3/10 | 5838.667 (n=3) | 4/10 | 5812.750 (n=4) |
| pituitary | CNN | 6/10 | 3071.833 (n=6) | 5/10 | 2782.800 (n=5) |
| pituitary | Swin3 | 4/10 | 2205.500 (n=4) | 5/10 | 3859.600 (n=5) |
| pituitary | Swin4 | 2/10 | 6692.000 (n=2) | 4/10 | 6917.250 (n=4) |
| Tümü | CNN | 29/40 | 2490.069 (n=29) | 21/40 | 2221.810 (n=21) |
| Tümü | Swin3 | 24/40 | 2366.208 (n=24) | 21/40 | 2458.000 (n=21) |
| Tümü | Swin4 | 15/40 | 5605.067 (n=15) | 18/40 | 5903.167 (n=18) |

| Gerçek sınıf | Kutu çifti | İki kutu var / n | Ortalama IoU |
|---|---|---:|---:|
| glioma | CNN/Swin3 | 1/10 | 0.007 (n=1) |
| glioma | CNN/Swin4 | 4/10 | 0.000 (n=4) |
| glioma | Swin3/Swin4 | 1/10 | 0.000 (n=1) |
| meningioma | CNN/Swin3 | 3/10 | 0.195 (n=3) |
| meningioma | CNN/Swin4 | 3/10 | 0.243 (n=3) |
| meningioma | Swin3/Swin4 | 3/10 | 0.417 (n=3) |
| notumor | CNN/Swin3 | 2/10 | 0.210 (n=2) |
| notumor | CNN/Swin4 | 3/10 | 0.222 (n=3) |
| notumor | Swin3/Swin4 | 3/10 | 0.181 (n=3) |
| pituitary | CNN/Swin3 | 3/10 | 0.000 (n=3) |
| pituitary | CNN/Swin4 | 2/10 | 0.000 (n=2) |
| pituitary | Swin3/Swin4 | 2/10 | 0.193 (n=2) |
| Tümü | CNN/Swin3 | 9/40 | 0.112 (n=9) |
| Tümü | CNN/Swin4 | 12/40 | 0.116 (n=12) |
| Tümü | Swin3/Swin4 | 9/40 | 0.242 (n=9) |

## j. Bölge koruma ve dolgu deneyi

Dolgular model girişindeki kanal ortalaması, 31×31 Gauss (σ=8) ve kafa dışı kanal medyanı. Yalnızca belirtilen bölge özgün haliyle korunur. Gauss dolgusu özgün girişten üretildiğinden silinen bölgenin düşük frekanslı bilgisini taşır. Alt bant son 56 satırdır ve beyin dokusu içerebilir. Marj hedef logit eksi hedef dışındaki en yüksek logittir. Düşüş özgün skor eksi müdahale skorudur ve negatif olabilir.

GA, 10.000 tekrar ve 20261119 tohumuyla hesaplanan %95 yüzdelik bootstrap aralığıdır. Tüm örneklemde sınıf başına mevcut geçerli kayıt sayısı kadar yeniden çekim yapıldı. Ön plan maskesi tüm görüntüyü kapladığında arka plan medyanı doğrulanmadı, o kayıtlar ilgili hesaplardan çıkarıldı ve geçerli n tabloda verildi. Aralıklar görüntü düzeyinde ve betimseldir, çoklu karşılaştırma düzeltmesi uygulanmadı. Hasta düzeyinde ayrım doğrulanmadı ve referans segmentasyon maskesi bulunmuyor.

| Sınıf (n) | Model | Korunan bölge | Dolgu | Softmax düşüşü (yüzde puan, %95 GA) | Logit marjı düşüşü (%95 GA) |
|---|---|---|---|---:|---:|
| glioma (10) | CNN | kafa dışı | ortalama | 16.02 [-0.30; 36.95] (n=10) | 2.41 [-1.09; 5.74] (n=10) |
| glioma (10) | CNN | kafa dışı | Gauss | 23.90 [9.32; 40.42] (n=10) | 4.02 [1.31; 6.68] (n=10) |
| glioma (10) | CNN | kafa dışı | arka plan medyanı | 65.17 [42.75; 84.99] (n=10) | 7.92 [4.15; 11.59] (n=10) |
| glioma (10) | CNN | alt bant | ortalama | 20.44 [9.11; 33.73] (n=10) | 4.92 [2.44; 7.74] (n=10) |
| glioma (10) | CNN | alt bant | Gauss | 33.70 [13.48; 54.20] (n=10) | 4.94 [1.95; 8.03] (n=10) |
| glioma (10) | CNN | alt bant | arka plan medyanı | 40.61 [22.25; 59.99] (n=10) | 5.83 [2.59; 9.17] (n=10) |
| glioma (10) | CNN | kafa dışı ∪ alt bant | ortalama | 13.51 [-1.58; 33.61] (n=10) | 0.65 [-2.88; 4.16] (n=10) |
| glioma (10) | CNN | kafa dışı ∪ alt bant | Gauss | 6.45 [1.44; 14.21] (n=10) | 2.60 [-0.05; 5.38] (n=10) |
| glioma (10) | CNN | kafa dışı ∪ alt bant | arka plan medyanı | 45.89 [24.98; 65.84] (n=10) | 5.86 [2.19; 9.50] (n=10) |
| glioma (10) | Swin-Tiny | kafa dışı | ortalama | 48.38 [31.85; 62.84] (n=10) | 7.57 [4.64; 10.16] (n=10) |
| glioma (10) | Swin-Tiny | kafa dışı | Gauss | 34.51 [21.75; 47.74] (n=10) | 6.75 [4.38; 9.06] (n=10) |
| glioma (10) | Swin-Tiny | kafa dışı | arka plan medyanı | 45.17 [22.75; 67.56] (n=10) | 7.69 [4.63; 10.44] (n=10) |
| glioma (10) | Swin-Tiny | alt bant | ortalama | 21.21 [8.86; 33.84] (n=10) | 5.93 [3.29; 8.25] (n=10) |
| glioma (10) | Swin-Tiny | alt bant | Gauss | 31.33 [21.89; 42.64] (n=10) | 6.73 [4.29; 8.91] (n=10) |
| glioma (10) | Swin-Tiny | alt bant | arka plan medyanı | 9.86 [2.82; 16.95] (n=10) | 5.14 [2.64; 7.38] (n=10) |
| glioma (10) | Swin-Tiny | kafa dışı ∪ alt bant | ortalama | 11.49 [-0.33; 24.99] (n=10) | 5.29 [2.67; 7.60] (n=10) |
| glioma (10) | Swin-Tiny | kafa dışı ∪ alt bant | Gauss | 9.32 [2.63; 17.97] (n=10) | 5.08 [2.96; 7.06] (n=10) |
| glioma (10) | Swin-Tiny | kafa dışı ∪ alt bant | arka plan medyanı | 6.08 [-3.71; 17.97] (n=10) | 4.40 [1.74; 6.79] (n=10) |
| meningioma (10) | CNN | kafa dışı | ortalama | 93.65 [86.30; 99.16] (n=10) | 12.36 [10.49; 13.97] (n=10) |
| meningioma (10) | CNN | kafa dışı | Gauss | 51.34 [29.56; 72.43] (n=10) | 5.59 [4.05; 6.94] (n=10) |
| meningioma (10) | CNN | kafa dışı | arka plan medyanı | 83.77 [68.95; 94.08] (n=10) | 8.63 [6.84; 10.42] (n=10) |
| meningioma (10) | CNN | alt bant | ortalama | 82.64 [63.07; 95.17] (n=10) | 7.98 [5.42; 10.34] (n=10) |
| meningioma (10) | CNN | alt bant | Gauss | 54.39 [30.38; 77.00] (n=10) | 5.90 [3.81; 8.09] (n=10) |
| meningioma (10) | CNN | alt bant | arka plan medyanı | 87.38 [69.00; 98.64] (n=10) | 9.74 [7.59; 11.62] (n=10) |
| meningioma (10) | CNN | kafa dışı ∪ alt bant | ortalama | 87.16 [67.47; 99.03] (n=10) | 14.11 [10.49; 17.34] (n=10) |
| meningioma (10) | CNN | kafa dışı ∪ alt bant | Gauss | 54.25 [29.71; 77.38] (n=10) | 6.06 [4.04; 8.15] (n=10) |
| meningioma (10) | CNN | kafa dışı ∪ alt bant | arka plan medyanı | 84.51 [65.97; 96.22] (n=10) | 8.75 [6.76; 10.60] (n=10) |
| meningioma (10) | Swin-Tiny | kafa dışı | ortalama | 82.11 [74.12; 88.73] (n=10) | 9.64 [7.59; 11.57] (n=10) |
| meningioma (10) | Swin-Tiny | kafa dışı | Gauss | 56.69 [42.05; 68.91] (n=10) | 8.27 [5.81; 10.45] (n=10) |
| meningioma (10) | Swin-Tiny | kafa dışı | arka plan medyanı | 92.38 [85.46; 96.82] (n=10) | 11.19 [8.92; 13.44] (n=10) |
| meningioma (10) | Swin-Tiny | alt bant | ortalama | 72.39 [55.03; 85.18] (n=10) | 9.06 [6.25; 11.50] (n=10) |
| meningioma (10) | Swin-Tiny | alt bant | Gauss | 49.95 [29.80; 65.57] (n=10) | 7.82 [4.91; 10.27] (n=10) |
| meningioma (10) | Swin-Tiny | alt bant | arka plan medyanı | 76.01 [63.19; 87.89] (n=10) | 9.88 [7.04; 12.41] (n=10) |
| meningioma (10) | Swin-Tiny | kafa dışı ∪ alt bant | ortalama | 80.64 [66.58; 92.01] (n=10) | 10.33 [7.53; 12.82] (n=10) |
| meningioma (10) | Swin-Tiny | kafa dışı ∪ alt bant | Gauss | 69.75 [45.35; 87.77] (n=10) | 9.34 [6.29; 11.81] (n=10) |
| meningioma (10) | Swin-Tiny | kafa dışı ∪ alt bant | arka plan medyanı | 84.09 [68.10; 96.50] (n=10) | 11.53 [8.35; 14.46] (n=10) |
| notumor (10) | CNN | kafa dışı | ortalama | 46.85 [25.44; 70.21] (n=10) | 8.86 [5.81; 12.00] (n=10) |
| notumor (10) | CNN | kafa dışı | Gauss | 51.93 [27.64; 76.29] (n=10) | 8.75 [6.95; 10.57] (n=10) |
| notumor (10) | CNN | kafa dışı | arka plan medyanı | 1.48 [-1.20; 4.20] (n=9) | 2.82 [-0.38; 6.02] (n=9) |
| notumor (10) | CNN | alt bant | ortalama | 20.40 [6.52; 37.72] (n=10) | 5.46 [3.77; 7.17] (n=10) |
| notumor (10) | CNN | alt bant | Gauss | 35.27 [12.63; 59.16] (n=10) | 6.35 [4.90; 7.80] (n=10) |
| notumor (10) | CNN | alt bant | arka plan medyanı | 3.26 [0.02; 7.23] (n=9) | 3.56 [1.75; 5.41] (n=9) |
| notumor (10) | CNN | kafa dışı ∪ alt bant | ortalama | 35.96 [13.26; 61.17] (n=10) | 6.87 [4.41; 10.02] (n=10) |
| notumor (10) | CNN | kafa dışı ∪ alt bant | Gauss | 39.78 [17.15; 63.91] (n=10) | 6.53 [4.94; 8.19] (n=10) |
| notumor (10) | CNN | kafa dışı ∪ alt bant | arka plan medyanı | -0.34 [-1.83; 0.61] (n=9) | 1.74 [-0.03; 3.58] (n=9) |
| notumor (10) | Swin-Tiny | kafa dışı | ortalama | 20.74 [9.83; 33.03] (n=10) | 7.63 [6.65; 8.62] (n=10) |
| notumor (10) | Swin-Tiny | kafa dışı | Gauss | 28.93 [13.78; 44.71] (n=10) | 8.31 [7.12; 9.52] (n=10) |
| notumor (10) | Swin-Tiny | kafa dışı | arka plan medyanı | 12.62 [1.12; 28.74] (n=9) | 5.98 [4.63; 7.67] (n=9) |
| notumor (10) | Swin-Tiny | alt bant | ortalama | 5.24 [3.27; 7.45] (n=10) | 6.38 [5.46; 7.30] (n=10) |
| notumor (10) | Swin-Tiny | alt bant | Gauss | 9.80 [5.99; 14.00] (n=10) | 7.34 [6.64; 8.00] (n=10) |
| notumor (10) | Swin-Tiny | alt bant | arka plan medyanı | 4.68 [2.00; 8.53] (n=9) | 6.13 [5.09; 7.25] (n=9) |
| notumor (10) | Swin-Tiny | kafa dışı ∪ alt bant | ortalama | 2.58 [0.45; 5.63] (n=10) | 4.81 [3.67; 6.01] (n=10) |
| notumor (10) | Swin-Tiny | kafa dışı ∪ alt bant | Gauss | 4.56 [1.59; 8.63] (n=10) | 6.05 [5.05; 6.99] (n=10) |
| notumor (10) | Swin-Tiny | kafa dışı ∪ alt bant | arka plan medyanı | 1.32 [0.13; 3.62] (n=9) | 3.74 [2.53; 5.21] (n=9) |
| pituitary (10) | CNN | kafa dışı | ortalama | 70.78 [56.78; 83.70] (n=10) | 9.19 [7.83; 10.81] (n=10) |
| pituitary (10) | CNN | kafa dışı | Gauss | 56.57 [39.27; 73.95] (n=10) | 8.17 [6.32; 9.98] (n=10) |
| pituitary (10) | CNN | kafa dışı | arka plan medyanı | 96.35 [94.56; 97.73] (n=10) | 11.48 [10.27; 12.68] (n=10) |
| pituitary (10) | CNN | alt bant | ortalama | 51.25 [29.11; 72.85] (n=10) | 7.79 [5.36; 10.11] (n=10) |
| pituitary (10) | CNN | alt bant | Gauss | 21.59 [1.33; 49.26] (n=10) | 4.92 [2.52; 7.47] (n=10) |
| pituitary (10) | CNN | alt bant | arka plan medyanı | 39.16 [18.48; 61.78] (n=10) | 7.25 [5.21; 9.31] (n=10) |
| pituitary (10) | CNN | kafa dışı ∪ alt bant | ortalama | 45.31 [22.97; 68.46] (n=10) | 7.69 [5.60; 9.96] (n=10) |
| pituitary (10) | CNN | kafa dışı ∪ alt bant | Gauss | 16.03 [0.36; 37.81] (n=10) | 3.95 [1.89; 6.13] (n=10) |
| pituitary (10) | CNN | kafa dışı ∪ alt bant | arka plan medyanı | 45.69 [21.48; 69.80] (n=10) | 7.43 [4.95; 9.72] (n=10) |
| pituitary (10) | Swin-Tiny | kafa dışı | ortalama | 89.21 [86.12; 92.01] (n=10) | 10.25 [8.46; 11.88] (n=10) |
| pituitary (10) | Swin-Tiny | kafa dışı | Gauss | 93.21 [90.97; 95.22] (n=10) | 10.84 [9.32; 12.22] (n=10) |
| pituitary (10) | Swin-Tiny | kafa dışı | arka plan medyanı | 93.35 [90.86; 95.71] (n=10) | 11.25 [9.55; 12.73] (n=10) |
| pituitary (10) | Swin-Tiny | alt bant | ortalama | 87.37 [82.69; 91.41] (n=10) | 9.87 [7.80; 11.64] (n=10) |
| pituitary (10) | Swin-Tiny | alt bant | Gauss | 91.81 [88.60; 94.49] (n=10) | 10.58 [8.61; 12.31] (n=10) |
| pituitary (10) | Swin-Tiny | alt bant | arka plan medyanı | 83.02 [77.17; 88.41] (n=10) | 9.50 [7.34; 11.40] (n=10) |
| pituitary (10) | Swin-Tiny | kafa dışı ∪ alt bant | ortalama | 81.53 [76.15; 86.12] (n=10) | 9.52 [7.79; 11.15] (n=10) |
| pituitary (10) | Swin-Tiny | kafa dışı ∪ alt bant | Gauss | 88.17 [80.48; 93.42] (n=10) | 10.46 [8.68; 12.08] (n=10) |
| pituitary (10) | Swin-Tiny | kafa dışı ∪ alt bant | arka plan medyanı | 84.33 [76.86; 90.69] (n=10) | 10.27 [8.39; 11.99] (n=10) |
| Tümü (40) | CNN | kafa dışı | ortalama | 56.82 [48.73; 65.27] (n=40) | 8.21 [6.91; 9.54] (n=40) |
| Tümü (40) | CNN | kafa dışı | Gauss | 45.93 [35.84; 56.01] (n=40) | 6.63 [5.62; 7.61] (n=40) |
| Tümü (40) | CNN | kafa dışı | arka plan medyanı | 63.24 [56.43; 69.35] (n=39) | 7.84 [6.49; 9.20] (n=39) |
| Tümü (40) | CNN | alt bant | ortalama | 43.68 [34.74; 52.16] (n=40) | 6.54 [5.39; 7.71] (n=40) |
| Tümü (40) | CNN | alt bant | Gauss | 36.23 [24.97; 47.84] (n=40) | 5.53 [4.35; 6.73] (n=40) |
| Tümü (40) | CNN | alt bant | arka plan medyanı | 43.61 [34.72; 52.14] (n=39) | 6.67 [5.45; 7.87] (n=39) |
| Tümü (40) | CNN | kafa dışı ∪ alt bant | ortalama | 45.48 [34.62; 56.22] (n=40) | 7.33 [5.78; 8.87] (n=40) |
| Tümü (40) | CNN | kafa dışı ∪ alt bant | Gauss | 29.13 [19.34; 39.23] (n=40) | 4.78 [3.70; 5.85] (n=40) |
| Tümü (40) | CNN | kafa dışı ∪ alt bant | arka plan medyanı | 45.07 [35.59; 53.93] (n=39) | 6.05 [4.74; 7.36] (n=39) |
| Tümü (40) | Swin-Tiny | kafa dışı | ortalama | 60.11 [54.79; 65.25] (n=40) | 8.77 [7.76; 9.75] (n=40) |
| Tümü (40) | Swin-Tiny | kafa dışı | Gauss | 53.34 [47.21; 59.37] (n=40) | 8.55 [7.58; 9.50] (n=40) |
| Tümü (40) | Swin-Tiny | kafa dışı | arka plan medyanı | 62.12 [55.37; 69.13] (n=39) | 9.11 [7.97; 10.18] (n=39) |
| Tümü (40) | Swin-Tiny | alt bant | ortalama | 46.55 [41.09; 51.38] (n=40) | 7.81 [6.70; 8.82] (n=40) |
| Tümü (40) | Swin-Tiny | alt bant | Gauss | 45.72 [39.77; 50.71] (n=40) | 8.12 [7.05; 9.10] (n=40) |
| Tümü (40) | Swin-Tiny | alt bant | arka plan medyanı | 44.39 [40.22; 48.24] (n=39) | 7.70 [6.55; 8.75] (n=39) |
| Tümü (40) | Swin-Tiny | kafa dışı ∪ alt bant | ortalama | 44.06 [39.16; 48.73] (n=40) | 7.49 [6.41; 8.52] (n=40) |
| Tümü (40) | Swin-Tiny | kafa dışı ∪ alt bant | Gauss | 42.95 [36.07; 48.45] (n=40) | 7.73 [6.68; 8.70] (n=40) |
| Tümü (40) | Swin-Tiny | kafa dışı ∪ alt bant | arka plan medyanı | 45.05 [39.82; 49.87] (n=39) | 7.58 [6.38; 8.71] (n=39) |

## Bulgular ve doğrulanmadı

- CNN ve Swin Aşama 4’te yumuşatma kapatıldığında notebook ve üretim haritaları ortalama Spearman ve SSIM’de üç ondalıkta 1,000’e ulaştı. Üretimdeki aug_smooth açıkken benzerlik düştü. CNN’de bir sabit harita çiftinde Spearman tanımsızdı. Hook, ReLU, normalizasyon ve yeniden boyutlandırmanın katkısını tek tek ayıran test doğrulanmadı.
- Swin Aşama 3’te her iki tarafta da kütüphane ve aug_smooth açıkken Spearman 0,514, SSIM 0,404 kaldı. Bu kontrollü karşılaştırmada esas fark notebook’un son bloğun norm1 katmanını ve üretimin aşama çıkışını kullanmasıdır.
- Aynı üretim kutu kuralı CNN ve Swin Aşama 3’te 21/40, Swin Aşama 4’te 18/40 kutu üretti. CNN/Swin ortak kutularının ortalama IoU değeri Aşama 3 için 0,112 (n=9), Aşama 4 için 0,116 (n=12) idi. Bu, iki tahminin kutu örtüşmesidir. Kutu konum doğruluğu doğrulanmadı.
- Glioma alt grubunda yalnız kafa dışı bölge korununca Swin’in hedef softmax skoru üç dolguda da ortalama 34,51–48,38 yüzde puan düştü. Birleşim korununca düşüşler daha küçüktü, ancak Swin logit marjı üç dolguda da 4,40–5,29 azaldı ve güven aralıkları sıfırı dışladı. CNN’de birleşim koşulunun sonucu dolguya duyarlıydı. Önceki küçük softmax düşüşü, hedefin rakiplerine karşı aynı marjı koruduğu anlamına gelmedi.
- Hedef, her modelin özgün görüntüde seçtiği sınıftır. Glioma satırları gerçek veri etiketine göre gruplandığından yanlış sınıflandırmalarda ölçülen skor başka bir sınıfın skorudur. Alt bant beyin dokusu içerebilir. Bu ölçümler kesin tümör dışı ipucu bağımlılığı sonucunu desteklemiyor.
- Doğrulanmadı. 23 numaralı notumor örneğinde arka plan maskesi boştu. İki model ve üç koruma koşulunda toplam altı medyan sonucu eksik kaydedildi. Hasta düzeyinde bağımsızlık ve anatomik tümör konumu doğrulanmadı.

## Dosyalar ve değişmezlik

Eklenen teslim dosyaları summary_ek.md ve metrics_ek.csv. Yedek içindeki run_ek.py ve report_ek.py koşu ve rapor tarifini içerir. Üretim ve model klasörlerindeki 46 dosyanın öncesi ve sonrası SHA-256 değerleri CSV’nin integrity kayıtlarında eşleşti. Önceki sonuç dosyalarının hash’leri de proje yedeğiyle karşılaştırıldı.
