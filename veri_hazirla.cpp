// Yardımcı program: Ödevlerde kullanılacak test görüntülerini bir kaynak görüntüden üretir.
//   0_gri_orijinal.png     : kaynak görüntünün gri seviye hali
//   1_dusuk_kontrast.png   : parlaklık aralığı 90..160'a sıkıştırılmış (düşük kontrastlı)
//   2_gauss_gurultulu.png  : Gauss gürültüsü eklenmiş (ortalama 0, sigma 20)
//   3_tuz_biber.png        : %5 tuz-biber gürültüsü (%2.5 siyah, %2.5 beyaz piksel)
//
// Not: Hocanın verdiği/bulduğunuz gerçek düşük kontrastlı veya gürültülü bir görüntü varsa
// onu doğrudan kullanabilirsiniz, bu program sadece elinizde hazır görüntü yokken test içindir.
//
// Kullanım: ./veri_hazirla <kaynak_goruntu> [cikti_dizini]   (varsayılan çıktı dizini: veri)

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <iostream>
#include <random>
#include <string>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Kullanim: " << argv[0] << " <kaynak_goruntu> [cikti_dizini]\n";
        return 1;
    }
    const fs::path outDir = (argc >= 3) ? fs::path(argv[2]) : fs::path("veri");
    fs::create_directories(outDir);

    cv::Mat src = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "Hata: goruntu okunamadi: " << argv[1] << "\n";
        return 1;
    }

    cv::Mat low(src.size(), CV_8UC1);
    cv::Mat gauss(src.size(), CV_8UC1);
    cv::Mat saltPepper(src.size(), CV_8UC1);

    std::mt19937 rng(42);  // sabit tohum: her çalıştırmada aynı sonuç
    std::normal_distribution<double> noise(0.0, 20.0);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            const int p = src.at<uchar>(y, x);

            // 0..255 -> 90..160
            low.at<uchar>(y, x) = cv::saturate_cast<uchar>(90.0 + p * 70.0 / 255.0);

            // Gauss gürültüsü
            gauss.at<uchar>(y, x) = cv::saturate_cast<uchar>(p + noise(rng));

            // Tuz-biber gürültüsü
            const double r = uni(rng);
            uchar v = static_cast<uchar>(p);
            if (r < 0.025) v = 0;
            else if (r < 0.05) v = 255;
            saltPepper.at<uchar>(y, x) = v;
        }
    }

    cv::imwrite((outDir / "0_gri_orijinal.png").string(), src);
    cv::imwrite((outDir / "1_dusuk_kontrast.png").string(), low);
    cv::imwrite((outDir / "2_gauss_gurultulu.png").string(), gauss);
    cv::imwrite((outDir / "3_tuz_biber.png").string(), saltPepper);

    std::cout << "Goruntu boyutu: " << src.cols << "x" << src.rows << "\n";
    std::cout << "Kaydedilenler (" << outDir.string() << "):\n"
              << "  0_gri_orijinal.png\n"
              << "  1_dusuk_kontrast.png   (Soru 1, 2, 3 icin)\n"
              << "  2_gauss_gurultulu.png  (Soru 4 icin)\n"
              << "  3_tuz_biber.png        (Soru 5 icin)\n";
    return 0;
}
