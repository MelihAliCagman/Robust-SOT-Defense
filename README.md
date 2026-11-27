# Robust-SOT-Defense

**Hedef takibinde görsel bozulmalara karşı dayanıklı izleme**
*Occlusion-robust single object tracking*

Derin Öğrenme dersi dönem projesi. **Geliştirme sürüyor.**

## Problem
Takip edilen bir hedef (araç, kişi) köprü, ağaç, bina ya da duman arkasına geçtiğinde hazır takipçiler hedefi kaybedebilir veya yanlış bir nesneye kayabilir. Bu proje, hazır bir derin takipçinin üstüne şu katmanı ekler:

1. Takipçi çıktısına güvenilip güvenilmeyeceğine karar vermek,
2. Güvenilmediğinde takipçiyi durdurup konumu Kalman filtresiyle tahmin etmek,
3. Hedef çıkınca onu yeniden bulup takipçiyi orada başlatmak.

## Nasıl çalışır?
```
Kare ──► [Donuk ViT takipçi] ──► kutu + güven skoru
                                   │
              [Güven kapısı: skor eşiği veya öğrenilmiş ağ]
                     │                          │
                  güvenilir                güvenilmez
                     │                          │
           Kalman güncelle            Takipçiyi durdur, Kalman tahmini,
           (yeşil kutu)               büyüyen pencerede hedefi ara (kırmızı kutu),
                                      bulunca takipçiyi yeniden başlat
```
- **Takipçi:** OpenCV ViT takipçisi, hazır ağırlıklarla (bu projede eğitilmez).
- **Öğrenilmiş kapı:** Son 8 karenin skor, skor farkı, kutu sıçraması, alan oranı ve şablon benzerliğini kullanan küçük bir tam bağlı ağ (scikit-learn MLP).
- **Yeniden yakalama:** Şablon eşleştirme (NCC), ardından takipçi skoruyla doğrulama.

## Güncel sonuçlar
Ön sonuçlardır; gerçek veri seti değerlendirmesi henüz yapılmadı.

**Sentetik test, kare düzeyinde "takipçi doğru mu?"** (70 videonun video bazında 42/14/14 eğitim/doğrulama/test bölmesi; eşikler doğrulamada seçildi)

| Kapı | AUROC | F1 |
|---|---|---|
| Skor eşiği | 0,847 | 0,886 |
| Öğrenilmiş ağ | 0,959 | 0,947 |

**Uçtan uca takip, 12 yeni sentetik video**

| Sistem | Success AUC | Precision@20px | Recovery |
|---|---|---|---|
| Takipçi tek başına | 0,634 | 0,713 | 0,58 |
| Skor eşiği + yeniden yakalama | 0,734 | 0,848 | 0,75 |
| Öğrenilmiş kapı + yeniden yakalama | 0,719 | 0,839 | 0,75 |

Öğrenilmiş ağ kare düzeyinde daha iyi, ancak uçtan uca ortalamada skor eşiğini geçemedi.

**Köprü videosu** (yapay zeka ile üretilmiş tek video; renkten çıkarılan **yaklaşık** etiketler, 218 kare)

| Sistem | Success AUC | Precision@20px |
|---|---|---|
| Takipçi tek başına | 0,350 | 0,405 |
| Skor eşiği 0,61 (sentetik doğrulamada seçilen) | 0,320 | 0,411 |
| Skor eşiği 0,45 (varsayılan) | 0,740 | 0,924 |
| Öğrenilmiş kapı | 0,645 | 0,903 |

Sonuç eşiğe çok duyarlı: sentetik veride seçilen eşik bu videoya uymuyor. Varsayılan 0,45 eşiğinin hangi veriye bakılarak belirlendiği bilinmediği için bağımsız bir seçim sayılmamalı. Tek videodan genel sonuç çıkarılamaz.

## Sınırlılıklar
- Kapanma sırasında tahmin kutusu hedefin gerisinde kalabilir.
- Yeniden yakalama piksel benzerliğine dayanır; görünüm değişirse veya benzer nesneler varsa zorlanabilir.
- Güven ağı yalnızca sentetik veriyle eğitildi.
- OpenCV takipçisi tepki haritası vermediği için PSR gibi ölçüler kullanılamıyor.

## Planlananlar
- OTB100 üzerinde video bazlı değerlendirme
- PyTorch ile zamansal güven modelleri ve karşılaştırmaları
- CNN tabanlı görünüm benzerliği ile yeniden yakalama
- Videoda hedef seçilebilen etkileşimli demo

## Kurulum
Python 3.10 veya üstü gerekir; mevcut sürüm CPU'da çalışır. Denenen ortam: Windows 11, Python 3.13.5, OpenCV 4.10.0, NumPy 2.4.2.

```
git clone https://github.com/MelihAliCagman/Robust-SOT-Defense.git
cd Robust-SOT-Defense
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Takipçi modeli depoda yoktur, [OpenCV Zoo](https://github.com/opencv/opencv_zoo)'dan indirilir:
```
curl.exe -L -o vit.onnx https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx
```
Dosya 714726 bayt olmalıdır.

Linux veya macOS'ta sanal ortam `source .venv/bin/activate` ile açılır ve `curl.exe` yerine `curl` yazılır. Colab için [main_colab.ipynb](main_colab.ipynb) dosyasına bakın.

## Kullanım
```
# Kurulum testi (sentetik sahne)
python scripts/selftest.py vit.onnx

# Kendi videonuzda: takipçi ve sistem yan yana (mp4, skor grafiği, csv)
python scripts/run_video.py --video video.mp4 --start 0 --box X Y W H --model vit.onnx --out out

# Eğitim verisi üretimi
python scripts/dump_features.py vit.onnx 70 feats_synth.npz

# Güven ağını eğitme ve ölçme (confidence_net.pkl ve gate_thresholds.npz üretir)
python scripts/train_eval_gate.py feats_synth.npz

# Skor eşiği ve öğrenilmiş kapıyı sentetik videolarda karşılaştırma (önceki adımın iki çıktısını okur)
python scripts/compare_gates.py vit.onnx --num_videos 12

# İsteğe bağlı: aynı iki adım, ek olarak kırmızı hedefli bir videoda da ölçer
python scripts/train_eval_gate.py feats_synth.npz vit.onnx video.mp4 START X Y W H
python scripts/compare_gates.py vit.onnx video.mp4 START X Y W H 12
```
- `X Y W H`: başlangıç karesindeki hedef kutusu (sol üst köşe, genişlik, yükseklik).
- `--video` bir görüntü klasörü de olabilir.
- Eğitim ve karşılaştırma video gerektirmez. İsteğe bağlı video bölümü yalnızca **kırmızı** hedefli videolarda anlamlıdır: etiket renkten yaklaşık olarak çıkarılır.
- Selftest için 27 Kasım 2026 ölçümü: takipçi 0,754 / skor + Kalman 0,421 / yeniden yakalamalı 0,828 (Success AUC). Farklı ortamlarda değerler değişebilir.

Videoda yeşil kutu takibi, kırmızı kutu kapanma durumundaki Kalman tahminini gösterir.

## OTB100 verisi
```
python scripts/download_otb.py
```
Resmî site erişilemediği için betik Internet Archive kopyalarını indirir; bazı sekansların arşiv kopyası yoktur. Veri bu depoya dahil değildir ve buradan dağıtılmaz. Veri setinin hakları sahiplerine aittir; kullanım koşulları bu projede doğrulanmamıştır.

## Depo yapısı
```
configs/       eşikler ve ayarlar
core/          takipçi sarmalayıcı, Kalman, özellikler, güven ağı, durum makinesi
utils/         kutu işlemleri, video okuma ve argümanları, çizim, sentetik sahne ve kapanma, renkten yaklaşık etiket
evaluation/    metrikler
scripts/       test, video çalıştırma, eğitim, karşılaştırma, veri indirme
```

## Yapay zeka kullanımı
Bu proje gerçekleştirilirken yapay zeka araçlarından destek alınmıştır. Köprü örnek videosu da yapay zeka ile üretilmiştir.
