// Ödev 3 - Soru 2: Histogram eşitleme (histogram equalization).
// Histogram çıkarma, kümülatif dağılım (CDF) ve eşitleme işlemleri HAZIR FONKSİYON KULLANILMADAN
// elle yazılmıştır.
//
// Adımlar:
//   1) 256 değerli histogram:      h[v]   = parlaklığı v olan piksel sayısı
//   2) Kümülatif dağılım (CDF):    cdf[v] = h[0] + h[1] + ... + h[v]
//   3) Eşitleme (dönüşüm tablosu): yeni(v) = round( (cdf[v] - cdfMin) * 255 / (N - cdfMin) )
//        N      : toplam piksel sayısı
//        cdfMin : sıfırdan büyük ilk CDF değeri (görüntüdeki en küçük parlaklığın piksel sayısı)
//   4) Her piksel dönüşüm tablosundan geçirilir.
//
// Kullanım: ./histogram_esitleme <dusuk_kontrastli_goruntu> [cikti_dizini]
//
// Derleme:
//   g++ -O2 odev3_2_histogram_esitleme.cpp -o histogram_esitleme.exe -std=c++17 \
//       $(pkg-config --cflags --libs opencv5)

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static const int BINS = 256;
using Hist = std::array<unsigned long long, BINS>;

// 1) Histogram: her parlaklık değerinden kaç piksel var
static Hist computeHistogram(const cv::Mat& img) {
    Hist h;
    h.fill(0);
    for (int y = 0; y < img.rows; ++y) {
        const uchar* s = img.ptr<uchar>(y);
        for (int x = 0; x < img.cols; ++x) ++h[s[x]];
    }
    return h;
}

// 2) Kümülatif dağılım: histogramın birikimli toplamı
static Hist computeCdf(const Hist& h) {
    Hist cdf;
    unsigned long long running = 0;
    for (int i = 0; i < BINS; ++i) {
        running += h[i];
        cdf[i] = running;
    }
    return cdf;
}

// Histogramı çubuk grafik olarak çizer (256 bin x 2 piksel genişlik, 300 piksel yükseklik)
static cv::Mat plotHistogram(const Hist& h) {
    const int H = 300, binW = 2;
    cv::Mat plot(H, BINS * binW, CV_8UC1, cv::Scalar(255));
    const unsigned long long maxCount = *std::max_element(h.begin(), h.end());
    for (int b = 0; b < BINS; ++b) {
        const int barH = maxCount ? static_cast<int>(static_cast<double>(h[b]) / maxCount * (H - 10)) : 0;
        for (int x = b * binW; x < (b + 1) * binW; ++x)
            for (int y = H - 1; y >= H - 1 - barH; --y) plot.at<uchar>(y, x) = 0;
    }
    return plot;
}

// CDF'yi çizgi grafik olarak çizer (0..1 arasına normalize edilmiş)
static cv::Mat plotCdf(const Hist& cdf, unsigned long long total) {
    const int H = 300, binW = 2;
    cv::Mat plot(H, BINS * binW, CV_8UC1, cv::Scalar(255));
    int prevY = H - 1;
    for (int b = 0; b < BINS; ++b) {
        const int curY = H - 1 - static_cast<int>(static_cast<double>(cdf[b]) / total * (H - 10));
        const int yTop = std::min(prevY, curY), yBot = std::max(prevY, curY);
        for (int x = b * binW; x < (b + 1) * binW; ++x)
            for (int y = yTop; y <= yBot; ++y) plot.at<uchar>(y, x) = 0;
        prevY = curY;
    }
    return plot;
}

static int countLevels(const Hist& h) {
    int c = 0;
    for (auto v : h) c += (v > 0) ? 1 : 0;
    return c;
}

static void minMaxFromHist(const Hist& h, int& mn, int& mx) {
    mn = -1;
    mx = -1;
    for (int i = 0; i < BINS; ++i) {
        if (h[i] > 0) {
            if (mn < 0) mn = i;
            mx = i;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Kullanim: " << argv[0] << " <dusuk_kontrastli_goruntu> [cikti_dizini]\n";
        return 1;
    }
    const fs::path outDir = (argc >= 3) ? fs::path(argv[2]) : fs::path("cikti_histogram_esitleme");
    fs::create_directories(outDir);

    cv::Mat src = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "Hata: goruntu okunamadi: " << argv[1] << "\n";
        return 1;
    }

    const unsigned long long N = static_cast<unsigned long long>(src.rows) * src.cols;

    // 1) Histogram, 2) CDF
    const Hist hist = computeHistogram(src);
    const Hist cdf = computeCdf(hist);

    // cdfMin: sıfırdan büyük ilk CDF değeri
    unsigned long long cdfMin = 0;
    for (int i = 0; i < BINS; ++i) {
        if (hist[i] > 0) {
            cdfMin = cdf[i];
            break;
        }
    }
    if (cdfMin == N) {
        std::cerr << "Hata: goruntu tek bir parlaklik degerinden olusuyor, esitlenemez.\n";
        return 1;
    }

    // 3) Dönüşüm tablosu (LUT)
    std::array<uchar, BINS> lut;
    for (int v = 0; v < BINS; ++v) {
        if (cdf[v] < cdfMin) {
            lut[v] = 0;  // bu parlaklıkta piksel yok, tablo girdisi kullanılmaz
        } else {
            const double mapped = static_cast<double>(cdf[v] - cdfMin) * 255.0 / static_cast<double>(N - cdfMin);
            lut[v] = static_cast<uchar>(mapped + 0.5);
        }
    }

    // 4) Eşitleme: her pikseli tablodan geçir
    cv::Mat out(src.size(), CV_8UC1);
    for (int y = 0; y < src.rows; ++y) {
        const uchar* s = src.ptr<uchar>(y);
        uchar* d = out.ptr<uchar>(y);
        for (int x = 0; x < src.cols; ++x) d[x] = lut[s[x]];
    }

    const Hist histOut = computeHistogram(out);
    const Hist cdfOut = computeCdf(histOut);

    int inMin, inMax, outMin, outMax;
    minMaxFromHist(hist, inMin, inMax);
    minMaxFromHist(histOut, outMin, outMax);

    std::cout << "Goruntu boyutu       : " << src.cols << "x" << src.rows << " (gri seviye), " << N << " piksel\n";
    std::cout << "cdfMin               : " << cdfMin << "\n";
    std::cout << "Esitlemeden once     : min = " << inMin << ", max = " << inMax
              << ", farkli seviye sayisi = " << countLevels(hist) << "\n";
    std::cout << "Esitlemeden sonra    : min = " << outMin << ", max = " << outMax
              << ", farkli seviye sayisi = " << countLevels(histOut) << "\n";
    std::cout << "CDF son degeri       : " << cdf[BINS - 1] << " (toplam piksel sayisina esit olmali: " << N << ")\n";

    // Doğrulama (sadece kontrol amaçlı): OpenCV'nin equalizeHist sonucu ile karşılaştır.
    // Ödevdeki histogram, CDF ve eşitleme bu fonksiyon kullanılmadan, yukarıdaki kodla yapıldı.
    cv::Mat ref;
    cv::equalizeHist(src, ref);
    std::cout << "Dogrulama: elle yazilan sonuc ile cv::equalizeHist arasindaki en buyuk fark = "
              << cv::norm(out, ref, cv::NORM_INF) << " (0 ya da 1 olmasi normal, yuvarlama farki)\n";

    // CSV: bin, histogram, cdf, eşitleme sonrası yeni değer
    {
        std::ofstream csv(outDir / "histogram_cdf.csv");
        csv << "parlaklik,histogram,cdf,yeni_deger\n";
        for (int v = 0; v < BINS; ++v) csv << v << "," << hist[v] << "," << cdf[v] << "," << int(lut[v]) << "\n";
    }

    // Grafikler: üst satır eşitlemeden önce (histogram | CDF), alt satır sonra
    cv::Mat topRow, bottomRow, panel;
    cv::hconcat(std::vector<cv::Mat>{plotHistogram(hist), plotCdf(cdf, N)}, topRow);
    cv::hconcat(std::vector<cv::Mat>{plotHistogram(histOut), plotCdf(cdfOut, N)}, bottomRow);
    cv::vconcat(std::vector<cv::Mat>{topRow, bottomRow}, panel);

    // Görüntülerin yan yana karşılaştırması (480 piksel genişlikte)
    auto small = [](const cv::Mat& m) {
        cv::Mat r;
        const double s = 480.0 / m.cols;
        cv::resize(m, r, cv::Size(), s, s, cv::INTER_AREA);
        return r;
    };
    cv::Mat montage;
    cv::hconcat(std::vector<cv::Mat>{small(src), small(out)}, montage);

    cv::imwrite((outDir / "1_dusuk_kontrast.png").string(), src);
    cv::imwrite((outDir / "2_esitlenmis.png").string(), out);
    cv::imwrite((outDir / "3_yan_yana.png").string(), montage);
    cv::imwrite((outDir / "4_histogram_cdf_grafikleri.png").string(), panel);

    std::cout << "\nKaydedilenler (" << outDir.string() << "):\n"
              << "  1_dusuk_kontrast.png            : girdi\n"
              << "  2_esitlenmis.png                : histogram esitleme sonucu (ciktı resim)\n"
              << "  3_yan_yana.png                  : soldan saga girdi | cikti\n"
              << "  4_histogram_cdf_grafikleri.png  : ust satir once (histogram | CDF), alt satir sonra\n"
              << "  histogram_cdf.csv               : her parlaklik icin histogram, CDF ve yeni deger\n";
    return 0;
}
