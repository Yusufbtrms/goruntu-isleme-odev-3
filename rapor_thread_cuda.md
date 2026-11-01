# Ödev 3: Multi-thread ve CUDA Sürümleri Raporu

Bu rapor, Ödev 3'teki beş görevin (doğrusal kontrast germe, histogram eşitleme, CLAHE, 3x3 ortalama filtre, 5x5 medyan filtre) çok iş parçacıklı (multi-thread) ve CUDA (GPU) sürümlerinin sonuçlarını özetler. Kodlar `odev3_thread.cpp` ve `odev3_cuda.cu` dosyalarındadır.

## 1. Yöntem

**Multi-thread (`odev3_thread.cpp`).** Her görev tek bir fonksiyon olarak yazıldı ve thread sayısı parametre alır; thread sayısı 1 olduğunda iş sırayla yapılır, böylece 1 thread ile N thread karşılaştırması aynı kodla yapılır. İş, görüntü satırlarına (CLAHE'de karelere) bölünür ve her thread kendi parçasını bağımsız işler.

- Kontrast germe: her thread kendi parçasının min/max değerini bulur, sonuçlar birleştirilir, ölçekleme satırlara bölünerek paralel yapılır.
- Histogram eşitleme: her thread kendi histogramını çıkarır, birleştirilir. CDF (256 eleman) sıralı, dönüşüm tablosunun uygulanması paralel.
- CLAHE: 64 karenin histogram, kırpma ve tablo hesapları thread'lere bölünür, ardından bilineer interpolasyon satırlara bölünerek paralel yapılır.
- Ortalama ve medyan filtre: satırlar thread'lere bölünür.

**CUDA (`odev3_cuda.cu`).** Beş görev için elle yazılmış GPU çekirdekleri kullanıldı (hazır GPU fonksiyonu yok): kontrast germede indirgeme çekirdeği, histogramda paylaşılan bellekte blok histogramı ve atomik toplama, CLAHE'de her kare için bir blok ve piksel başına interpolasyon, filtrelerde piksel başına bir GPU thread'i. Derleme: `nvcc -O2 -std=c++17 --fmad=false -arch=native ... -Xcompiler "-O2,-pthread"`. `--fmad=false`, GPU'nun ondalıklı hesapları CPU ile aynı sırada ve yuvarlamayla yapmasını sağlar. `-Xcompiler -O2` olmadan CPU kısmı optimize edilmez ve CPU süreleri 4-9 kat uzun çıkar; karşılaştırma için şarttır.

**Doğrulama.** Her thread sayısının sonucu 1 thread sonucuyla, GPU sonuçları da CPU tek thread sonucuyla piksel piksel karşılaştırıldı.

## 2. Ortamlar

- **Thread deneyi:** kendi bilgisayarım, 16 mantıksal işlemci, 1600x1598 gri görüntüler, 10 tekrarın ortalaması (medyan için 3).
- **CUDA deneyi:** Google Colab, Tesla T4 (hesaplama yeteneği 7.5), 2 vCPU. CPU süreleri bu ortamdaki CPU'dan alındı (CPU 5 tekrar, medyan için 1, GPU 200 tekrar).

İki deneyin CPU süreleri farklı makinelerden gelir ve birbiriyle doğrudan karşılaştırılmamalıdır.

## 3. Sonuçlar

### 3.1 Multi-thread (kendi bilgisayarım)

Çalışma süresi (ms):

| Görev | 1 thread | 2 thread | 4 thread | 8 thread | Sonuç aynı mı |
|---|---|---|---|---|---|
| 1. Kontrast germe | 2.81 | 1.91 | 1.30 | 1.67 | Evet |
| 2. Histogram eşitleme | 2.82 | 2.13 | 1.47 | 1.77 | Evet |
| 3. CLAHE | 9.55 | 5.69 | 4.15 | 3.23 | Evet |
| 4. Ortalama filtre 3x3 | 9.97 | 5.49 | 3.96 | 2.48 | Evet |
| 5. Medyan filtre 5x5 | 559.91 | 362.78 | 242.26 | 153.83 | Evet |

Hızlanma (1 thread süresi / N thread süresi):

| Görev | 2 thread | 4 thread | 8 thread |
|---|---|---|---|
| 1. Kontrast germe | 1.47x | 2.16x | 1.69x |
| 2. Histogram eşitleme | 1.33x | 1.92x | 1.59x |
| 3. CLAHE | 1.68x | 2.30x | 2.96x |
| 4. Ortalama filtre 3x3 | 1.81x | 2.52x | 4.01x |
| 5. Medyan filtre 5x5 | 1.54x | 2.31x | 3.64x |

### 3.2 CUDA (Colab, Tesla T4)

Çalışma süresi (ms). "GPU çekirdek" görüntü zaten GPU belleğindeyken yalnız hesaplamadır, "GPU toplam" görüntüyü GPU'ya yüklemeyi ve sonucu geri indirmeyi içerir.

| Görev | CPU 1 thread | CPU 2 thread | GPU çekirdek | GPU toplam | Sonuç CPU ile aynı mı |
|---|---|---|---|---|---|
| 1. Kontrast germe | 5.587 | 5.616 | 0.262 | 1.505 | Birebir aynı |
| 2. Histogram eşitleme | 4.895 | 4.787 | 0.169 | 1.377 | Birebir aynı |
| 3. CLAHE | 23.660 | 21.857 | 0.205 | 1.380 | Birebir aynı |
| 4. Ortalama filtre 3x3 | 21.346 | 19.604 | 0.093 | 1.293 | Birebir aynı |
| 5. Medyan filtre 5x5 | 893.086 | 568.540 | 5.377 | 5.648 | Birebir aynı |

Hızlanma (CPU 1 thread süresi / GPU süresi):

| Görev | GPU çekirdek | GPU toplam |
|---|---|---|
| 1. Kontrast germe | 21.33x | 3.71x |
| 2. Histogram eşitleme | 29.00x | 3.56x |
| 3. CLAHE | 115.65x | 17.14x |
| 4. Ortalama filtre 3x3 | 229.18x | 16.50x |
| 5. Medyan filtre 5x5 | 166.10x | 158.11x |

## 4. Yorum

**Multi-thread.** Hızlanma hiçbir görevde thread sayısıyla doğrusal artmadı: 4 thread'de 1.92x ile 2.52x arası, 8 thread'de 1.59x ile 4.01x arası kaldı. Kontrast germe ve histogram eşitleme gibi 1-3 ms süren hafif işlerde 8 thread, 4 thread'den daha yavaş çıktı (2.16x'ten 1.69x'e ve 1.92x'ten 1.59x'e). Olası açıklama, thread açıp kapatma maliyetinin ve bellek erişim sınırının bu kısa işlerde kazancın önemli kısmını yemesidir; bu ölçülmedi. En ağır iş olan medyan filtre 560 ms'den 154 ms'ye indi (8 thread'de 3.64x), ancak beklenenin aksine 3x3 ortalama filtre (4.01x) daha iyi ölçeklendi. Nedeni bu verilerle belirlenemedi; makinedeki fiziksel çekirdek sayısının mantıksal işlemci sayısından az olması olası bir etkendir ama doğrulanmadı.

**CUDA.** GPU çekirdeklerinin kendisi çok hızlıdır (0.09-0.34 ms, medyan 5.4 ms), ancak toplam süre görüntüyü CPU ile GPU arasında taşımak yüzünden sabit olarak yaklaşık 1.2 ms artar (yaklaşık 5 MB gidiş-dönüş). Bu nedenle:

- Hafif işlerde (kontrast germe, histogram eşitleme) çekirdek 21-29 kat hızlı olsa da, aktarım dahil toplam kazanç yalnızca 3.6-3.7 kattır. Hesaplama süresi (0.2-0.3 ms) aktarımın yanında küçük kalır.
- Hesaplama yoğun işlerde (CLAHE, medyan) kazanç büyüktür: CLAHE 17 kat, medyan 158 kat (893 ms'den 5.6 ms'ye).
- Verinin GPU'da birden çok işlemden geçeceği bir hat kurulursa aktarım bir kez ödenir ve çekirdek hızlanmaları gerçeğe daha yakın olur.

**Kaba bir karşılaştırma.** Kendi bilgisayarımdaki en iyi çok thread'li CPU süresini (kontrast germe 1.30, histogram 1.47, CLAHE 3.23, ortalama 2.48, medyan 153.83 ms) Colab GPU toplam süresiyle oranlarsam GPU yaklaşık 0.86x (kontrast germe, yani daha yavaş), 1.07x, 2.3x, 1.9x ve 27x olur. İki farklı ortamın karışımı olduğu için bu oranlar sadece yön gösterir; ana çıkarım, hafif görevlerde iyi bir çok thread'li CPU'nun aktarım maliyetli GPU'ya yaklaştığı, medyan gibi ağır görevlerde GPU'nun açık fark attığıdır.

**Doğruluk.** Beş görevin beşinde GPU sonucu CPU tek thread sonucuyla birebir aynıdır (en büyük fark 0, farklı piksel 0).

## 5. Sınırlılıklar

- Her ölçüm tek bir görüntü boyutunda (1600x1598) ve tek çalıştırmayla alındı. Çalıştırmalar arası değişim (önceki deneylerde yüzde 25'e kadar) ölçülmedi.
- CPU ve GPU karşılaştırması Colab'ın 2 vCPU'lu makinesinde yapıldı; kendi bilgisayarındaki CPU çok daha hızlıdır (örneğin medyan 1 thread: 560 ms ve Colab'da 893 ms).
- Medyan GPU süresinde çekirdek (5.377 ms) ile toplam (5.648 ms) arasındaki fark (0.27 ms), diğer görevlerdeki aktarım maliyetinden (yaklaşık 1.2 ms) küçüktür; medyanın GPU sürelerinde ±1-2 ms belirsizlik vardır.
- CLAHE'de parametreler sabit (8x8 kare, clip limit 2.0); GPU çekirdeklerinde kare başına işlem seri bir iş parçacığıyla yapılır, daha iyi paralelleştirme denenmedi.
- Elle yazılan CPU ve GPU kodları OpenCV'nin hazır fonksiyonlarıyla süre karşılaştırmasına tabi tutulmadı.
