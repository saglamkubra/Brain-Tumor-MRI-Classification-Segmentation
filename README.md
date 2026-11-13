# Beyin MR Görüntülerinde Tümör Sınıflandırma ve Segmentasyon

**Brain Tumor MRI Classification and Segmentation**

## 1. Projenin Amacı

Bu proje, beyin MR görüntülerinde tümör varlığının ve türünün derin öğrenme yöntemleriyle sınıflandırılmasını, tümörün yaklaşık konumunun belirlenmesini ve Segment Anything Model (SAM) kullanılarak ilgili bölgenin segmentasyonunu amaçlamaktadır.

Projede üç farklı sınıflandırma mimarisi karşılaştırılmaktadır:

- Convolutional Neural Network (CNN)
- Vision Transformer (ViT)
- Swin Transformer

Yaklaşık konum belirleme amacıyla Grad-CAM ve uygun açıklanabilir yapay zekâ yöntemleri kullanılmaktadır. Elde edilen konum önerileri, SAM için yönlendirme olarak değerlendirilmektedir.

**Not:** Bu çalışma eğitim ve araştırma amaçlıdır. Klinik tanı sistemi olarak kullanılmamalıdır.

## 2. Kullanılan Veri Seti

**Veri seti:** Brain Tumor MRI Dataset

**Kaynak:** https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Veri setinde dört sınıf bulunmaktadır:

- **Glioma:** Glioma tümörü
- **Meningioma:** Meningioma tümörü
- **Pituitary:** Hipofiz tümörü
- **No Tumor:** Tümör bulunmayan görüntüler

### Veri Dağılımı

| Veri Bölümü | Görüntü Sayısı |
|---|---:|
| Eğitim | 4.480 |
| Doğrulama | 1.120 |
| Test | 1.600 |
| **Toplam** | **7.200** |

Eğitim ve doğrulama kümeleri, veri setinin `Training` bölümünden oluşturulmuştur. `Testing` bölümü bağımsız test kümesi olarak kullanılmıştır.

Veri seti büyük boyutlu olduğundan GitHub deposuna yüklenmemektedir.

## 3. Kullanılan Teknolojiler

### CNN

Evrişimsel sinir ağı mimarisi kullanılarak MR görüntülerinin dört sınıfa ayrılması hedeflenmektedir.

### Vision Transformer (ViT)

Görüntülerin parçalara ayrılarak Transformer mimarisiyle işlenmesi ve sınıflandırılması amaçlanmaktadır.

### Swin Transformer

Kaydırılmış pencere tabanlı Transformer mimarisiyle MR görüntüleri sınıflandırılmaktadır.

### Grad-CAM

Modelin sınıflandırma kararını etkileyen görüntü bölgelerini görselleştirmek amacıyla kullanılmaktadır.

Grad-CAM çıktıları kesin tümör sınırları olarak değerlendirilmemektedir.

### Segment Anything Model (SAM)

Önceden eğitilmiş SAM ViT-B modeli, görüntü ve konum kutusu girdileriyle segmentasyon maskeleri üretmek için kullanılmaktadır.

İki yaklaşım incelenmektedir:

1. Grad-CAM'den otomatik oluşturulan konum kutuları.
2. Görüntü üzerinde elle belirlenen konum kutuları.

Bu yöntemler karşılaştırılarak konum yönlendirmesinin segmentasyon sonucuna etkisi araştırılmaktadır.

## 4. Model Eğitimi ve Değerlendirme

Modellerin değerlendirilmesinde aşağıdaki metrikler kullanılmaktadır:

- Accuracy (Doğruluk)
- Precision (Kesinlik)
- Recall (Duyarlılık)
- F1-score
- Confusion Matrix (Karmaşıklık Matrisi)

### Swin Transformer Test Sonuçları

| Metrik | Sonuç |
|---|---:|
| Accuracy | %93,25 |
| Precision (Macro) | %93,88 |
| Recall (Macro) | %93,25 |
| F1-score (Macro) | %93,19 |

Swin Transformer, önceden eğitilmiş ağırlıklar kullanılarak dört sınıflı sınıflandırma görevine uyarlanmıştır.

CNN ve ViT sonuçları tamamlandığında karşılaştırma tablosuna eklenecektir.

## 5. Yaklaşık Konum Belirleme ve Segmentasyon

Swin Transformer'ın sınıflandırma sonuçları üzerinden Grad-CAM aktivasyon haritaları elde edilmiştir.

Bu haritalardan yaklaşık konum kutuları oluşturulmuş ve SAM modeline verilmiştir.

İlk deneylerde otomatik kutuların bazı görüntülerde gereğinden geniş olduğu ve SAM'ın tümör yerine büyük anatomik bölgeleri maskeleyebildiği gözlemlenmiştir.

Elle belirlenen kutularla gerçekleştirilen deneylerde ise bazı örneklerde lezyona görsel olarak daha uygun maskeler elde edilmiştir.

Bu sonuçlar, sınıflandırma başarısının doğrudan konumlandırma başarısı anlamına gelmediğini göstermektedir.

Veri setinde uzmanlar tarafından hazırlanmış tümör segmentasyon maskeleri bulunmadığından, mevcut deneylerde segmentasyon doğruluğu Dice veya IoU gibi metriklerle doğrulanamamaktadır.

## 6. Geliştirme Ortamı

Proje Google Colab ortamında Python kullanılarak geliştirilmektedir.

Başlıca kütüphaneler:

- PyTorch
- timm
- torchvision
- pytorch-grad-cam
- segment-anything
- OpenCV
- NumPy
- pandas
- Matplotlib
- scikit-learn

## 7. Kurulum ve Çalıştırma

1. Veri setini Kaggle bağlantısından indirin.
2. Google Drive üzerinde bir proje klasörü oluşturun.
3. Veri setini bu klasöre yükleyin.
4. İlgili Google Colab notebook dosyasını açın.
5. Colab çalışma zamanında GPU seçin.
6. Notebook içerisindeki Google Drive dosya yollarını kendi ortamınıza göre düzenleyin.
7. Kod hücrelerini sırasıyla çalıştırın.

Eğitilmiş model ağırlıkları ve büyük veri dosyaları GitHub deposunda tutulmamaktadır.

Notebook dosyaları ve ayrıntılı çalıştırma talimatları proje geliştirme sürecinde depoya eklenecektir.

## 8. Projenin Mevcut Durumu

- [x] Veri setinin hazırlanması
- [x] Eğitim, doğrulama ve test bölümlerinin oluşturulması
- [x] Swin Transformer modelinin eğitilmesi
- [x] Swin Transformer test sonuçlarının hesaplanması
- [x] Grad-CAM ile yaklaşık konum belirleme deneyleri
- [x] SAM ile ilk segmentasyon deneyleri
- [x] Otomatik ve manuel konum yönlendirmelerinin incelenmesi
- [ ] CNN sonuçlarının entegrasyonu
- [ ] ViT sonuçlarının entegrasyonu
- [ ] Modellerin karşılaştırılması
- [ ] Nihai rapor ve sunum

## 9. Proje Ekibi

Bu proje iki kişilik bir ekip tarafından geliştirilmektedir.

**Görev dağılımı:**

- **Ekip üyesi 1:** CNN ve ViT geliştirme, eğitim ve değerlendirme.
- **Ekip üyesi 2:** Swin Transformer, Grad-CAM ve SAM geliştirme, deney ve değerlendirme.

Raporlama, sonuçların karşılaştırılması ve sunum çalışmaları ortak yürütülmektedir.

## 10. Teslim Tarihi

**24 Kasım 2026**

---

**Bu proje bir üniversite dersi kapsamında eğitim ve araştırma amacıyla hazırlanmıştır.**
