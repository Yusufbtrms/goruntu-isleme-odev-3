// Ödev 3 - Soru 1: Düşük kontrastlı, tek kanallı görüntüde min/max bulup parlaklığı
//                  [0, 255] aralığına DOĞRUSAL ölçekleme (hazır ölçekleme fonksiyonu yok).
//
// Formül:  yeni = (eski - min) * 255 / (max - min)
//
// Kullanım: ./kontrast_germe <dusuk_kontrastli_goruntu> [cikti_dizini]
//
// Derleme:
//   g++ -O2 odev3_1_kontrast_germe.cpp -o kontrast_germe.exe -std=c++17 \
//       $(pkg-config --cflags --libs opencv5)

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Kullanim: " << argv[0] << " <dusuk_kontrastli_goruntu> [cikti_dizini]\n";
        return 1;
    }
    const fs::path outDir = (argc >= 3) ? fs::path(argv[2]) : fs::path("cikti_kontrast_germe");
    fs::create_directories(outDir);

    // Tek kanallı (gri seviye) olarak aç
    cv::Mat src = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "Hata: goruntu okunamadi: " << argv[1] << "\n";
        return 1;
    }

    // 1) Minimum ve maksimum değeri elle bul
    int minV = 255, maxV = 0;
    for (int y = 0; y < src.rows; ++y) {
        const uchar* s = src.ptr<uchar>(y);
        for (int x = 0; x < src.cols; ++x) {
            const int v = s[x];
            if (v < minV) minV = v;
            if (v > maxV) maxV = v;
        }
    }

    if (maxV == minV) {
        std::cerr << "Hata: goruntu tek bir parlaklik degerinden olusuyor (" << minV
                  << "), olceklenemez.\n";
        return 1;
    }

    // 2) Doğrusal ölçekleme: [min, max] -> [0, 255]
    const double scale = 255.0 / (maxV - minV);
    cv::Mat out(src.size(), CV_8UC1);
    for (int y = 0; y < src.rows; ++y) {
        const uchar* s = src.ptr<uchar>(y);
        uchar* d = out.ptr<uchar>(y);
        for (int x = 0; x < src.cols; ++x) {
            d[x] = static_cast<uchar>((s[x] - minV) * scale + 0.5);  // +0.5: en yakın tam sayıya yuvarla
        }
    }

    // Sonucun min/max değerini de elle kontrol et
    int outMin = 255, outMax = 0;
    for (int y = 0; y < out.rows; ++y) {
        const uchar* d = out.ptr<uchar>(y);
        for (int x = 0; x < out.cols; ++x) {
            if (d[x] < outMin) outMin = d[x];
            if (d[x] > outMax) outMax = d[x];
        }
    }

    std::cout << "Goruntu boyutu     : " << src.cols << "x" << src.rows << " (gri seviye)\n";
    std::cout << "Olceklemeden once  : min = " << minV << ", max = " << maxV
              << ", aralik = " << (maxV - minV) << "\n";
    std::cout << "Olceklemeden sonra : min = " << outMin << ", max = " << outMax << "\n";

    // Doğrulama (sadece kontrol amaçlı): OpenCV'nin normalize sonucu ile karşılaştır.
    // Ödevdeki ölçekleme bu fonksiyon kullanılmadan, yukarıdaki elle yazılmış döngüyle yapıldı.
    cv::Mat ref;
    cv::normalize(src, ref, 0, 255, cv::NORM_MINMAX, CV_8U);
    std::cout << "Dogrulama: elle yazilan sonuc ile cv::normalize arasindaki en buyuk fark = "
              << cv::norm(out, ref, cv::NORM_INF) << " (0 ya da 1 olmasi normal, yuvarlama farki)\n";

    // Görsel karşılaştırma için yan yana görüntü (her biri 480 piksel genişlikte)
    auto small = [](const cv::Mat& m) {
        cv::Mat r;
        const double s = 480.0 / m.cols;
        cv::resize(m, r, cv::Size(), s, s, cv::INTER_AREA);
        return r;
    };
    std::vector<cv::Mat> row = {small(src), small(out)};
    cv::Mat montage;
    cv::hconcat(row, montage);

    cv::imwrite((outDir / "1_dusuk_kontrast.png").string(), src);
    cv::imwrite((outDir / "2_olceklenmis.png").string(), out);
    cv::imwrite((outDir / "3_yan_yana.png").string(), montage);

    std::cout << "\nKaydedilenler (" << outDir.string() << "):\n"
              << "  1_dusuk_kontrast.png : girdi\n"
              << "  2_olceklenmis.png    : ciktilar [0, 255] araligina gerilmis\n"
              << "  3_yan_yana.png       : soldan saga girdi | cikti\n";
    return 0;
}
